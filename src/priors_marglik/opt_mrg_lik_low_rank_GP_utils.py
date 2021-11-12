import os
import socket
import datetime
import torch
import numpy as np
import torch.nn.functional as F
import torch.autograd.functional as AF
import torch.autograd as autograd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from deep_image_prior import gaussian_log_prob, tv_loss
from dataset import extract_tafo_as_matrix
from linearized_laplace import submatrix_low_rank_GP_lin_model_prior_cov
import tensorly as tl
tl.set_backend('pytorch')

def marginal_lik_PredCP_linear_update(
    cfg, 
    block_priors, 
    Jac_x, 
    recon
    ):

    _, list_model_prior_cov = submatrix_low_rank_GP_lin_model_prior_cov(block_priors, Jac_x)
    expected_tv = []
    for cov in list_model_prior_cov:
        succed = False
        cnt = 0
        while not succed: 
            try: 
                dist = \
                    torch.distributions.multivariate_normal.MultivariateNormal(loc=recon.flatten().to(block_priors.store_device),
                        covariance_matrix=cov)
                succed = True 
            except: 
                cov[np.diag_indices(cov.shape[0])] += 1e-4 # cov.diag().detach().mean() / 1000
                cnt += 1
            assert cnt < 100

        samples = dist.rsample((100, ))
        expected_tv.append(tv_loss(samples).mean(dim=0))

    log_det_list = []
    for i in range(block_priors.num_params):

        assert block_priors.log_lengthscales[i].grad == None \
            or block_priors.log_lengthscales[i].grad == 0
        assert block_priors.log_variances[i].grad == None \
            or block_priors.log_variances[i].grad == 0

        first_derivative = autograd.grad(expected_tv[i],
                block_priors.log_lengthscales[i], retain_graph=True,
                create_graph=True)[0]  # delta_k_d/delta_\ell_d
        first_derivative_log_variances = autograd.grad(expected_tv[i],
                block_priors.log_variances[i], allow_unused=True,
                retain_graph=True)[0]
        log_det = first_derivative.abs().log()
        log_det_list.append(log_det.detach())
        second_derivative = autograd.grad(log_det, block_priors.log_lengthscales[i], retain_graph=True)[0]
        second_derivative_log_variances = autograd.grad(log_det, block_priors.log_variances[i], allow_unused=True)[0]

        first_derivative = first_derivative.detach()
        second_derivative = second_derivative.detach()
        block_priors.priors[i].zero_grad()

        block_priors.log_lengthscales[i].grad = -(-first_derivative
                * cfg.mrglik.optim.scaling_fct * cfg.mrglik.optim.scl_fct_gamma
                * cfg.mrglik.optim.gamma + second_derivative) 

        block_priors.log_variances[i].grad = -(-first_derivative_log_variances \
            * cfg.mrglik.optim.scaling_fct * cfg.mrglik.optim.scl_fct_gamma \
            * cfg.mrglik.optim.gamma + second_derivative_log_variances )
    
    loss = cfg.mrglik.optim.scaling_fct * cfg.mrglik.optim.scl_fct_gamma \
        * cfg.mrglik.optim.gamma * torch.stack(expected_tv).sum().detach() \
        - torch.stack(log_det_list).sum().detach()

    return loss

def post_hess_log_det_y_space(
    block_priors,
    Jac_y,
    log_noise_model_variance, 
    ):

    log_prior_det_inv = \
        -block_priors.get_net_log_det_cov_mat()
    Kyy = \
        block_priors.matrix_prior_cov_mul(Jac_y) @ Jac_y.transpose(1, 0)
    (sign, kernel_det) = torch.linalg.slogdet( 
        torch.eye(Jac_y.shape[0]).to(log_noise_model_variance.device) * torch.exp(log_noise_model_variance)
        + Kyy)

    assert sign > 0
    
    return log_prior_det_inv - log_noise_model_variance * Jac_y.shape[0] + kernel_det

def optim_marginal_lik_low_rank_GP(
    cfg,
    observation,
    recon,
    block_priors,
    Jac_x,
    Jac_y,
    ray_trafos, 
    lin_weights = None,
    comment=''
    ):


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'mrglik_opt' + comment

    logdir = os.path.join(
        './', comment + '_' +  current_time + '_' + socket.gethostname())

    writer = SummaryWriter(log_dir=logdir)

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    log_noise_model_variance_y = torch.nn.Parameter(
        torch.zeros(1, device=device)
        )

    trafo, _, _, _ = extract_tafo_as_matrix(ray_trafos)

    recon_proj = (trafo @ recon.flatten()).to(device)
    observation = observation.to(device).flatten()

    optimizer = \
        torch.optim.Adam([{'params': block_priors.log_lengthscales, 'lr': cfg.mrglik.optim.lr},
                          {'params': block_priors.log_variances, 'lr': cfg.mrglik.optim.lr},
                          {'params': log_noise_model_variance_y, 'lr': cfg.mrglik.optim.lr}]
                        )

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik. opt') as pbar:
        for i in pbar:

            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predCP:
                predCP_loss = \
                    marginal_lik_PredCP_linear_update(cfg, block_priors, Jac_x, torch.from_numpy(recon).to(device))
            else: 
                predCP_loss = torch.zeros(1)

            posterior_hess_log_det_y_space = post_hess_log_det_y_space(
                block_priors,
                Jac_y,
                log_noise_model_variance_y, 
                )

            reconstruction_log_lik = gaussian_log_prob(
                observation,
                recon_proj,
                torch.exp(log_noise_model_variance_y)
                )

            if cfg.linearize_weights and lin_weights is not None:
                weight_prior_log_prob = \
                    block_priors.get_net_prior_log_prob_lin_weights(lin_weights)
            else:
                weight_prior_log_prob = \
                    block_priors.get_net_prior_log_prob()
            
            loss = -(reconstruction_log_lik + weight_prior_log_prob - 0.5 * posterior_hess_log_det_y_space)

            loss.backward()
            optimizer.step()

            for k in range(block_priors.num_params):
                writer.add_scalar('lengthscale_{}'.format(k), 
                                torch.exp(block_priors.log_lengthscales[k]).item(), i)
                writer.add_scalar('variance_{}'.format(k), 
                                torch.exp(block_priors.log_variances[k]).item(), i)

            writer.add_scalar('negative_MAP_MLL', loss.item() + predCP_loss.item(), i)
            writer.add_scalar('negative_MLL', loss.item(), i)
            writer.add_scalar('reconstruction_log_lik', reconstruction_log_lik.item(), i)
            writer.add_scalar('posterior_hess_log_det_y_space', posterior_hess_log_det_y_space.item(), i)
            writer.add_scalar('weight_prior_log_prob', weight_prior_log_prob.item(), i)
            writer.add_scalar('predCP', -predCP_loss.item(), i)
            writer.add_scalar('log_lik_variance_y', torch.exp(log_noise_model_variance_y).item(), i)


    return torch.exp(log_noise_model_variance_y).detach()
