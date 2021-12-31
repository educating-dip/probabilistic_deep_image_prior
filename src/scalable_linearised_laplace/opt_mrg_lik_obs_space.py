import os
import socket
import datetime
import torch
import numpy as np
from torch.linalg import cholesky
import torch.autograd as autograd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import diag_gaussian_log_prob, tv_loss
from linearized_laplace import submatrix_image_space_lin_model_prior_cov
from linearized_weights import get_weight_block_vec
from scalable_linearised_laplace import compute_approx_log_det_grad, vec_weight_prior_cov_mul, get_diag_prior_cov_obs_mat

def marginal_lik_log_det_update(bayesianized_model, grads, step_size, return_loss=False):
    parameters = (
            bayesianized_model.gp_log_lengthscales +
            bayesianized_model.gp_log_variances +
            bayesianized_model.normal_log_variances)

    for param in parameters:
        if param.grad is None:
            param.grad = -(step_size * 0.5) * grads[param]
        else:
            param.grad -= (step_size * 0.5) * grads[param]

    if return_loss:
        raise NotImplementedError

def optim_marginal_lik_low_rank(
    cfg,
    observation,
    recon,
    ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules,
    jacobi_vector=None,
    comment=''
    ):


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'mrglik_opt' + comment
    logdir = os.path.join(
        './', comment + '_' +  current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=logdir)
    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    recon, proj_recon = recon
    recon = recon.to(device).flatten()
    proj_recon = proj_recon.to(device).flatten()
    observation_shape = observation.shape[1:]
    observation = observation.to(device).flatten()

    weight_vec = get_weight_block_vec(bayesianized_model.get_all_modules_under_prior())[None]

    log_noise_model_variance_obs = torch.nn.Parameter(
        torch.zeros(1, device=device),
    )
    optimizer = \
        torch.optim.Adam([{'params': bayesianized_model.gp_log_lengthscales, 'lr': cfg.mrglik.optim.lr},
                          {'params': bayesianized_model.gp_log_variances, 'lr': cfg.mrglik.optim.lr},
                          {'params': bayesianized_model.normal_log_variances, 'lr': cfg.mrglik.optim.lr},
                          {'params': log_noise_model_variance_obs, 'lr': cfg.mrglik.optim.lr}]
                        )

    if jacobi_vector is None:
        jacobi_vector = get_diag_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, replace_by_identity=True).detach()

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik.opt') as pbar:
        for i in pbar:

            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predcp:
                raise NotImplementedError
            else: 
                predcp_loss = torch.zeros(1)

            # update grads for post_hess_log_det
            grads = compute_approx_log_det_grad(
                    ray_trafos, filtbackproj,
                    bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules,
                    log_noise_model_variance_obs.detach(),
                    cfg.mrglik.impl.vec_batch_size, side_length=observation_shape,
                    use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp, jacobi_vector=jacobi_vector)
            marginal_lik_log_det_update(bayesianized_model, grads, step_size=cfg.mrglik.optim.lr)

            obs_log_density = (torch.sum(observation - proj_recon) ** 2) / torch.exp(log_noise_model_variance_obs)  # 0.5 * sigma_y^-2 ||y_delta - A f(theta^*)||_2^2

            weight_prior_norm = (vec_weight_prior_cov_mul(bayesianized_model, weight_vec, return_inverse=True) @ weight_vec.T).squeeze()
            
            loss = -0.5 * (obs_log_density + weight_prior_norm)

            loss.backward()
            optimizer.step()

            for k, gp_log_lengthscale in enumerate(bayesianized_model.gp_log_lengthscales):
                writer.add_scalar('gp_lengthscale_{}'.format(k),
                                torch.exp(gp_log_lengthscale).item(), i)
            for k, gp_log_variance in enumerate(bayesianized_model.gp_log_variances):
                writer.add_scalar('gp_variance_{}'.format(k),
                                torch.exp(gp_log_variance).item(), i)
            for k, normal_log_variance in enumerate(bayesianized_model.normal_log_variances):
                writer.add_scalar('normal_variance_{}'.format(k),
                                torch.exp(normal_log_variance).item(), i)

            writer.add_scalar('negative_MAP_MLL', loss.item() + predcp_loss.item(), i)
            writer.add_scalar('negative_MLL', loss.item(), i)
            writer.add_scalar('obs_log_density', obs_log_density.item(), i)
            writer.add_scalar('weight_prior_norm', weight_prior_norm.item(), i)
            writer.add_scalar('predcp', -predcp_loss.item(), i)
            writer.add_scalar('noise_model_variance_obs', torch.exp(log_noise_model_variance_obs).item(), i)

    return torch.exp(log_noise_model_variance_obs).detach()
