import os
import socket
import datetime
import torch
import numpy as np
import torch.nn as nn
import torch.autograd as autograd
from copy import deepcopy
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import tv_loss, list_norm_layers, set_all_weights, get_weight_vec, PSNR

def get_weight_grad_vec(model, norm_layers):
    ws = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            ws.append(param.grad.flatten())
    return torch.cat(ws)

def finite_diff_JvP(x, model, vec, eps=None):

    assert len(vec.shape) == 1
    model.eval()
    with torch.no_grad():
        norm_layers = list_norm_layers(model)
        map_weights = get_weight_vec(model, norm_layers)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max)  / v_max

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights(model, norm_layers, w_plus)
        f_plus = model(x)[0]
        #         del w_plus

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights(model, norm_layers, w_minus)
        f_minus = model(x)[0]
        #         del w_minus

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights(model, norm_layers, map_weights)
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


def weights_linearization(cfg, x, observation, ground_truth, reconstructor, ray_trafos):

    norm_layers = list_norm_layers(reconstructor.model)
    map_weights = get_weight_vec(reconstructor.model, norm_layers)
    w_init = torch.zeros_like(map_weights)
    ray_trafo_module = ray_trafos['ray_trafo_module'].to(reconstructor.device)
    ray_trafo_module_adj = ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)
    observation = observation.to(reconstructor.device)
    ground_truth = ground_truth.to(reconstructor.device)

    reconstructor.model.eval()
    lin_w_fd = nn.Parameter(w_init.clone()).to(reconstructor.device)
    optimizer = torch.optim.Adam([lin_w_fd], **{'lr': cfg.mrglik.linearized_weights.lr}, weight_decay=0)

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'lin_weights_optim'

    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)

    writer = SummaryWriter(log_dir=logdir)
    loss_vec_fd, psnr = [], []

    with tqdm(range(cfg.mrglik.linearized_weights.iterations)) as pbar:
        for i in pbar:

            # get projection vector
            lin_pred = finite_diff_JvP(x, reconstructor.model, lin_w_fd).detach()
            loss = torch.nn.functional.mse_loss(ray_trafo_module(lin_pred), observation.to(reconstructor.device)).detach() + cfg.net.optim.gamma * tv_loss(lin_pred)
            v = log_homoGauss_grad(ray_trafo_module(lin_pred), observation, ray_trafo_module_adj).flatten() + cfg.net.optim.gamma * tv_loss_grad(lin_pred).flatten()
            optimizer.zero_grad()
            reconstructor.model.zero_grad()
            to_grad = (reconstructor.model(x)[0].flatten() * v)
            to_grad.sum().backward()
            lin_w_fd.grad = get_weight_grad_vec(reconstructor.model, norm_layers) + cfg.mrglik.linearized_weights.wd * lin_w_fd.detach() # take the weights
            optimizer.step()

            loss_vec_fd.append(loss.detach().item())
            psnr.append(PSNR(ground_truth.cpu().numpy(), lin_pred.cpu().numpy()))
            pbar.set_postfix({'psnr': psnr[-1]})
            writer.add_scalar('loss', loss_vec_fd[-1], i)
            writer.add_scalar('psnr', psnr[-1], i)


# def test_optim(reconstructor, filtbackproj, store_device):
#
#     reconstructor.model.eval()
#     kappa = [[], [], [], []]
#     for lengthscale_init in np.logspace(-2, 2, 100):
#         block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, lengthscale_init)
#         lengthscales = [param for param in block_priors.parameters() if param.requires_grad]
#         expected_tv = block_priors.get_expected_TV_loss(filtbackproj)
#         dist = torch.distributions.exponential.Exponential(torch.ones(1, device=store_device))
#         for i in range(len(expected_tv)):
#             log_pi = dist.log_prob(expected_tv[i])
#             first_derivative = autograd.grad(expected_tv[i], lengthscales[i])[0] # delta_k/delta_\ell
#             log_det = first_derivative.abs().log()
#             kappa[i].append((log_pi + log_det).detach().cpu().item())
#     return kappa, np.logspace(-2, 2, 100)
