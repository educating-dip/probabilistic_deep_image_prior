import os
import socket
import datetime
import torch
import numpy as np
import torch.nn as nn
from torch.nn import DataParallel
from collections.abc import Iterable
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import tv_loss, PSNR
from linearized_laplace import agregate_flatten_weight_grad

def finite_diff_JvP(x, model, vec, eps=None):

    assert len(vec.shape) == 1
    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec(model)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block(model, w_plus)
        f_plus = model(x)[0]

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block(model, w_minus)
        f_minus = model(x)[0]

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block(model, map_weights)
        return JvP

def log_homoGauss_grad(mean, y, ray_trafo_module_adj, prec=1):
    return - (prec * ray_trafo_module_adj(y - mean))

def tv_loss_grad(x):

    assert x.shape[-1] == x.shape[-2]

    sign_diff_x =  torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((1, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y =  torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((1, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y

def set_all_weights(model, norm_layers, weights):
    """ set all NN weights """
    assert not isinstance(model, DataParallel)
    n_weights_all = 0
    for name, param in model.named_parameters():
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            n_weights = param.numel()
            param.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(param))
            n_weights_all += n_weights

def set_all_weights_block(model, weights, include_block=['down', 'up']):
    """ set all NN weights """
    assert not isinstance(model, DataParallel)
    n_weights_all = 0
    for sect_name in include_block:
        group_blocks = getattr(model, sect_name)
        if isinstance(group_blocks, Iterable):
            for (_, block) in enumerate(group_blocks):
                for layer in block.conv:
                    if isinstance(layer, torch.nn.Conv2d):
                        n_weights = layer.weight.numel()
                        layer.weight.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
                        n_weights_all += n_weights

def get_weight_block_vec(model, include_block=['down', 'up']):
    ws = []
    for sect_name in include_block:
        group_blocks = getattr(model, sect_name)
        if isinstance(group_blocks, Iterable):
            for (_, block) in enumerate(group_blocks):
                for layer in block.conv:
                    if isinstance(layer, torch.nn.Conv2d):
                        ws.append(layer.weight.flatten())
    return torch.cat(ws)

def get_weight_vec(model, norm_layers):
    ws = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            ws.append(param.flatten())
    return torch.cat(ws)

def list_norm_layers(model):

    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """
    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, torch.nn.GroupNorm) or isinstance(module,
                torch.nn.BatchNorm2d) or isinstance(module, torch.nn.InstanceNorm2d):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers

def weights_linearization(cfg, x, observation, ground_truth, reconstructor, ray_trafos):

    x = x.to(reconstructor.device)
    observation = observation.to(reconstructor.device)
    ground_truth = ground_truth.to(reconstructor.device)
    map_weights = get_weight_block_vec(reconstructor.model).detach()
    ray_trafo_module = ray_trafos['ray_trafo_module'].to(reconstructor.device)
    ray_trafo_module_adj = ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)
    lin_w_fd = nn.Parameter(torch.zeros_like(map_weights).clone()).to(reconstructor.device)
    optimizer = torch.optim.Adam([lin_w_fd], **{'lr': cfg.lin_params.lr}, weight_decay=0)
    
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'lin_weights_optim'
    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)
    writer = SummaryWriter(log_dir=logdir)
    loss_vec_fd, psnr = [], []

    reconstructor.model.eval()
    with tqdm(range(cfg.lin_params.iterations)) as pbar:
        for i in pbar:
            lin_pred = finite_diff_JvP(x, reconstructor.model, lin_w_fd).detach()
            loss = torch.nn.functional.mse_loss(ray_trafo_module(lin_pred), observation.to(reconstructor.device)).detach() \
                + cfg.net.optim.gamma * tv_loss(lin_pred)
            v = log_homoGauss_grad(ray_trafo_module(lin_pred), observation, ray_trafo_module_adj).flatten() \
                + cfg.net.optim.gamma * tv_loss_grad(lin_pred).flatten()
            optimizer.zero_grad()
            reconstructor.model.zero_grad()
            to_grad = reconstructor.model(x)[0].flatten() * v
            to_grad.sum().backward()
            lin_w_fd.grad = agregate_flatten_weight_grad(reconstructor.model) + cfg.lin_params.wd * lin_w_fd.detach()
            optimizer.step()

            loss_vec_fd.append(loss.detach().item())
            psnr.append(PSNR(lin_pred.cpu().numpy(), ground_truth.cpu().numpy()))

            if i % 100 == 0: 
                print(torch.sum((lin_w_fd)**2))
                print(torch.sum((map_weights)**2))

            pbar.set_postfix({'psnr': psnr[-1]})
            writer.add_scalar('loss', loss_vec_fd[-1], i)
            writer.add_scalar('psnr', psnr[-1], i)

    return lin_w_fd.detach(), lin_pred.detach()