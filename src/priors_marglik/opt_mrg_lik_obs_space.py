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

def marginal_lik_predcp_linear_update(
    cfg, 
    block_priors, 
    Jac_x, 
    recon, 
    obs_shape
    ):

    _, model_prior_cov_list = submatrix_image_space_lin_model_prior_cov(block_priors, Jac_x)
    if not cfg.mrglik.priors.include_normal_priors:
        model_prior_cov_list = model_prior_cov_list[:len(block_priors.gp_priors)]
    expected_tv = []
    for cov_ff in model_prior_cov_list:
        succed = False
        cnt = 0
        while not succed:
            try: 
                dist = \
                    torch.distributions.multivariate_normal.MultivariateNormal(
                        loc=recon,
                        scale_tril=cholesky(cov_ff)
                    )
                succed = True 
            except: 
                cov_ff[np.diag_indices(cov_ff.shape[0])] += 1e-6
                cnt += 1
            assert cnt < 1000

        samples = dist.rsample((cfg.mrglik.optim.tv_samples, ))
        expected_tv.append(tv_loss(samples.view(-1, *recon.shape)) / cfg.mrglik.optim.tv_samples)

    log_det_list = []
    for i, gp_prior in enumerate(block_priors.gp_priors):

        assert gp_prior.cov.log_lengthscale.grad == None \
            or gp_prior.cov.log_lengthscale.grad == 0
        assert gp_prior.cov.log_variance.grad == None \
            or gp_prior.cov.log_variance.grad == 0

        first_derivative = autograd.grad(expected_tv[i],
                gp_prior.cov.log_lengthscale, retain_graph=True,
                create_graph=True)[0]
        first_derivative_log_variances = autograd.grad(expected_tv[i],
                gp_prior.cov.log_variance, allow_unused=True,
                retain_graph=True)[0]
        log_det = first_derivative.abs().log()
        log_det_list.append(log_det.detach())
        second_derivative = autograd.grad(log_det, gp_prior.cov.log_lengthscale, retain_graph=True)[0]
        second_derivative_log_variances = autograd.grad(log_det, gp_prior.cov.log_variance, allow_unused=True)[0]
        first_derivative = first_derivative.detach()
        second_derivative = second_derivative.detach()
        gp_prior.zero_grad()
        scaling_fct = cfg.mrglik.optim.scaling_fct * obs_shape * cfg.mrglik.optim.gamma
        gp_prior.cov.log_lengthscale.grad = -(-first_derivative + second_derivative) * scaling_fct
        gp_prior.cov.log_variance.grad = -(-first_derivative_log_variances + second_derivative_log_variances ) * scaling_fct

    for normal_prior in block_priors.normal_priors:
        pass  # TODO
    
    loss = scaling_fct * (torch.stack(expected_tv).sum().detach() - torch.stack(log_det_list).sum().detach())
    return loss

def post_hess_log_det_obs_space(
    block_priors,
    Jac_obs,
    log_noise_model_variance_obs, 
    ):

    log_prior_det_inv = -block_priors.get_net_log_det_cov_mat()
    Kyy = block_priors.matrix_prior_cov_mul(Jac_obs) @ Jac_obs.T
    sign, kernel_det = torch.linalg.slogdet( 
        torch.eye(Jac_obs.shape[0], device = log_noise_model_variance_obs.device) * torch.exp(log_noise_model_variance_obs)
        + Kyy)
    assert sign > 0
    
    return log_prior_det_inv - log_noise_model_variance_obs * Jac_obs.shape[0] + kernel_det

def optim_marginal_lik_low_rank(
    cfg,
    observation,
    recon,
    block_priors,
    Jac,
    Jac_obs,
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
    observation = observation.to(device).flatten()

    log_noise_model_variance_obs = torch.nn.Parameter(
        torch.zeros(1, device=device)
    )
    optimizer = \
        torch.optim.Adam([{'params': block_priors.gp_log_lengthscales, 'lr': cfg.mrglik.optim.lr},
                          {'params': block_priors.gp_log_variances, 'lr': cfg.mrglik.optim.lr},
                          {'params': block_priors.normal_log_variances, 'lr': cfg.mrglik.optim.lr},
                          {'params': log_noise_model_variance_obs, 'lr': cfg.mrglik.optim.lr}]
                        )

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik.opt', miniters=cfg.mrglik.optim.iterations//100) as pbar:
        for i in pbar:

            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predcp:
                predcp_loss = \
                    marginal_lik_predcp_linear_update(
                        cfg, 
                        block_priors,
                        Jac,
                        recon,
                        observation.numel()
                    )
            else: 
                predcp_loss = torch.zeros(1)

            post_hess_log_det = post_hess_log_det_obs_space(
                block_priors,
                Jac_obs,
                log_noise_model_variance_obs, 
            )

            obs_log_density = diag_gaussian_log_prob(
                observation,
                proj_recon,
                torch.exp(log_noise_model_variance_obs)
            )

            if block_priors.lin_weights is not None: 
                weight_prior_log_prob = \
                    block_priors.get_net_prior_log_prob_linearized_weights()
            else:
                weight_prior_log_prob = \
                    block_priors.get_net_prior_log_prob()

            loss = -(obs_log_density + weight_prior_log_prob - 0.5 * post_hess_log_det)

            loss.backward()
            optimizer.step()

            for k, gp_log_lengthscale in enumerate(block_priors.gp_log_lengthscales):
                writer.add_scalar('gp_lengthscale_{}'.format(k),
                                torch.exp(gp_log_lengthscale).item(), i)
            for k, gp_log_variance in enumerate(block_priors.gp_log_variances):
                writer.add_scalar('gp_variance_{}'.format(k),
                                torch.exp(gp_log_variance).item(), i)
            for k, normal_log_variance in enumerate(block_priors.normal_log_variances):
                writer.add_scalar('normal_variance_{}'.format(k),
                                torch.exp(normal_log_variance).item(), i)

            writer.add_scalar('negative_MAP_MLL', loss.item() + predcp_loss.item(), i)
            writer.add_scalar('negative_MLL', loss.item(), i)
            writer.add_scalar('obs_log_density', obs_log_density.item(), i)
            writer.add_scalar('posterior_hess_log_det_obs_space', post_hess_log_det.item(), i)
            writer.add_scalar('weight_prior_log_prob', weight_prior_log_prob.item(), i)
            writer.add_scalar('predcp', -predcp_loss.item(), i)
            writer.add_scalar('noise_model_variance_obs', torch.exp(log_noise_model_variance_obs).item(), i)


    return torch.exp(log_noise_model_variance_obs).detach()
