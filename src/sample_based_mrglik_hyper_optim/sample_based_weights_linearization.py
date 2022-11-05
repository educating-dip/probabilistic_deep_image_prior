import os
import socket
import datetime
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from deep_image_prior import  PSNR
from linearized_laplace import agregate_flatten_weight_grad
from scalable_linearised_laplace import get_weight_block_vec, get_fwAD_model, fwAD_JvP
from copy import deepcopy 

def assemble_prior_diag(num_weights_per_prior_var, prior_vars):

    n_weights = np.sum(num_weights_per_prior_var)
    prior_var_per_weight = torch.ones(n_weights, device=prior_vars.device)
    prior_var_per_weight_splits = []
    for prior_var, split in zip(prior_vars, torch.split(prior_var_per_weight, num_weights_per_prior_var)):
        split = split * prior_var
        prior_var_per_weight_splits.append(split)
    prior_var_per_weight = torch.cat(prior_var_per_weight_splits)
    return prior_var_per_weight

def log_homoGauss_grad(mean, y, ray_trafo_module_adj, prec=1):
    return - (prec * ray_trafo_module_adj(y - mean))

def weights_linearization(
    cfg,
    bayesianized_model, model, all_modules_under_prior,
    filtbackproj, observation, ground_truth, 
    ray_trafos, 
    prior_variance_vec, 
    noise_model_variance_obs, 
    initial_map_weights=None
    ):


    # converting computation to single precision (this is to speed up)
    filtbackproj, observation, ground_truth = filtbackproj.float(), observation.float(),  ground_truth.float()
    ray_trafo_module = deepcopy(ray_trafos['ray_trafo_module']).float()
    ray_trafo_module_adj = deepcopy(ray_trafos['ray_trafo_module_adj']).float()
    prior_variance_vec = prior_variance_vec.float()
    noise_model_variance_obs = deepcopy(noise_model_variance_obs).float()
    
    bayesianized_model.float()
    model.float()

    store_device = bayesianized_model.store_device

    map_weights = initial_map_weights.detach().clone() if initial_map_weights is not None else torch.zeros_like(
            get_weight_block_vec(all_modules_under_prior).detach()
        )
    
    fwAD_model, fwAD_module_mapping = get_fwAD_model(model, return_module_mapping=True, share_parameters=True)
    fwAD_modules = [fwAD_module_mapping[m] for m in all_modules_under_prior]

    lin_w_fd = nn.Parameter(map_weights).to(store_device)
    optimizer = torch.optim.Adam([lin_w_fd], **{'lr': cfg.sample_based_mrglik.weights_linearization.lr}, weight_decay=0)
    psnr = []

    if len(prior_variance_vec) != 1:
        num_weights_per_prior_var_list = bayesianized_model.ref_num_params_per_modules_under_gp_priors + bayesianized_model.ref_num_params_per_modules_under_normal_priors
        prior_variance_vec = assemble_prior_diag(num_weights_per_prior_var_list, prior_variance_vec).float()
    else:
        pass

    with tqdm(range(cfg.sample_based_mrglik.weights_linearization.iterations), miniters=cfg.sample_based_mrglik.weights_linearization.iterations//1000) as pbar:
        for _ in pbar:

            fd_vector = lin_w_fd
            lin_pred = fwAD_JvP(filtbackproj, fwAD_model, fd_vector, fwAD_modules, pre_activation=True, saturation_safety=False).detach()
            
            _ = ( 1 / noise_model_variance_obs) * torch.nn.functional.mse_loss(
                ray_trafo_module(lin_pred), observation, reduction='sum'
                ) + ( (prior_variance_vec**-1) * lin_w_fd.pow(2) ).sum()

            v = log_homoGauss_grad(
                ray_trafo_module(lin_pred), observation, ray_trafo_module_adj, prec=noise_model_variance_obs**-1).flatten() 
          
            optimizer.zero_grad()
            model.zero_grad()
            to_grad = model(filtbackproj)[1].flatten() * v
            to_grad.sum().backward()

            lin_w_fd.grad = ( 
                agregate_flatten_weight_grad(all_modules_under_prior) + (prior_variance_vec**-1) * lin_w_fd.detach() 
                ) / observation.numel()
            optimizer.step()

            psnr.append(PSNR(lin_pred.detach().cpu().numpy(), ground_truth.cpu().numpy()))
            pbar.set_description('psnr={:.1f}'.format(psnr[-1]), refresh=False)

        mse_loss = ( (ray_trafo_module(lin_pred) - observation.to(bayesianized_model.store_device) )**2 ).sum()
    
    # converting back to double precision 
    if cfg.use_double:
        bayesianized_model.double()
        model.double()

        lin_w_fd = lin_w_fd.double()
        lin_pred = lin_pred.double()
        mse_loss = mse_loss.double()

    return lin_w_fd.detach(), lin_pred.detach(), mse_loss.detach()