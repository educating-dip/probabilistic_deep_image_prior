import os
import socket
import datetime
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import tv_loss, PSNR
from linearized_laplace import agregate_flatten_weight_grad

# jacobian vector product w.r.t. the `weight` parameters of `modules`
def finite_diff_JvP(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 1
    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec(modules)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block(modules, w_plus)
        f_plus = model(x, saturation_safety=False)[1] # pre-activation 

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block(modules, w_minus)
        f_minus = model(x, saturation_safety=False)[1] # pre-activation

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block(modules, map_weights)
        return JvP

def log_homoGauss_grad(mean, y, ray_trafo_module_adj, prec=1):
    return - (prec * ray_trafo_module_adj(y - mean))

def tv_loss_grad(x):

    assert x.shape[-1] == x.shape[-2]

    sign_diff_x = torch.sign(torch.diff(-x, n=1, dim=-1))
    pad = torch.zeros((1, 1, x.shape[-2], 1), device = x.device)
    diff_x_pad = torch.cat([pad, sign_diff_x, pad], dim=-1)
    grad_tv_x = torch.diff(diff_x_pad, n=1, dim=-1)
    sign_diff_y = torch.sign(torch.diff(-x, n=1, dim=-2))
    pad = torch.zeros((1, 1, 1, x.shape[-1]), device = x.device)
    diff_y_pad = torch.cat([pad, sign_diff_y, pad], dim=-2)
    grad_tv_y = torch.diff(diff_y_pad, n=1, dim=-2)
    
    return grad_tv_x + grad_tv_y

def set_all_weights_block(modules, weights):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        n_weights = layer.weight.numel()
        layer.weight.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
        n_weights_all += n_weights

def get_weight_block_vec(modules):
    ws = []
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        ws.append(layer.weight.flatten())
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

def weights_linearization(cfg, bayesianised_model, filtbackproj, observation, ground_truth, reconstructor, ray_trafos):

    filtbackproj = filtbackproj.to(reconstructor.device)
    observation = observation.to(reconstructor.device)
    ground_truth = ground_truth.to(reconstructor.device)

    all_modules_under_prior = bayesianised_model.get_all_modules_under_prior()
    map_weights = get_weight_block_vec(all_modules_under_prior).detach()
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

    # if cfg.mrglik.impl.use_fwAD_for_jvp: TODO 

    #     from scalable_linearised_laplace import get_fwAD_model
    #     from scalable_linearised_laplace import fwAD_JvP
    #     fwAD_model, fwAD_module_mapping = get_fwAD_model(reconstructor.model, return_module_mapping=True, use_copy='share_parameters')
    #     fwAD_modules = [fwAD_module_mapping[m] for m in all_modules_under_prior]

    reconstructor.model.eval()
    with tqdm(range(cfg.lin_params.iterations)) as pbar:
        for i in pbar:

            lin_pred = finite_diff_JvP(filtbackproj, reconstructor.model, lin_w_fd, all_modules_under_prior).detach()            
             
            if cfg.net.arch.use_sigmoid:
                lin_pred = lin_pred.sigmoid()

            loss = torch.nn.functional.mse_loss(ray_trafo_module(lin_pred), observation.to(reconstructor.device)) \
                + cfg.net.optim.gamma * tv_loss(lin_pred)

            v = 2 / observation.numel() * log_homoGauss_grad(ray_trafo_module(lin_pred), observation, ray_trafo_module_adj).flatten() \
                + cfg.net.optim.gamma * tv_loss_grad(lin_pred).flatten() 

            if cfg.net.arch.use_sigmoid:
                v = v * lin_pred.flatten() * (1 - lin_pred.flatten())
            
            optimizer.zero_grad()
            reconstructor.model.zero_grad()
            to_grad = reconstructor.model(filtbackproj)[1].flatten() * v
            to_grad.sum().backward()
            lin_w_fd.grad = agregate_flatten_weight_grad(all_modules_under_prior) + cfg.lin_params.wd * lin_w_fd.detach()
            optimizer.step()

            loss_vec_fd.append(loss.detach().item())
            psnr.append(PSNR(lin_pred.detach().cpu().numpy(), ground_truth.cpu().numpy()))
            pbar.set_postfix({'psnr': psnr[-1]})
            writer.add_scalar('loss', loss_vec_fd[-1], i)
            writer.add_scalar('psnr', psnr[-1], i)

    return lin_w_fd.detach(), lin_pred.detach()