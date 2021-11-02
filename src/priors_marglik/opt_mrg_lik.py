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
from deep_image_prior import gaussian_log_prob
import tensorly as tl
tl.set_backend('pytorch')


def marginal_lik_PredCP_update(
    cfg,
    reconstructor,
    filtbackproj,
    block_priors
    ):

    expected_tv = block_priors.get_expected_TV_loss(filtbackproj)
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
        second_derivative = autograd.grad(log_det,
                block_priors.log_lengthscales[i], retain_graph=True)[0]
        second_derivative_log_variances = autograd.grad(log_det,
                block_priors.log_variances[i], allow_unused=True)[0]

        first_derivative = first_derivative.detach()
        second_derivative = second_derivative.detach()
        block_priors.priors[i].zero_grad()

        block_priors.log_lengthscales[i].grad = -(-first_derivative * cfg.mrglik.optim.scaling_fct  + second_derivative)
        block_priors.log_variances[i].grad = \
            -first_derivative_log_variances * cfg.mrglik.optim.scaling_fct + second_derivative_log_variances


def est_lik_hess(obs, recon, recon_no_sigmoid, filtbackproj, ray_trafos, lik_variance=1, mode='sigmoid_exact'):

    forward_operator_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
    forward_operator_mat = forward_operator_mat.view(-1, ray_trafos['space'].shape[0]**2)
    forward_operator_mat_adj = forward_operator_mat.t()
    A_t_A = forward_operator_mat_adj @ forward_operator_mat

    if mode == 'linear':
        noise_mat = ( 1 / lik_variance ) * A_t_A

    elif mode == 'sigmoid_exact':
        loss_grad = - forward_operator_mat_adj @ (obs.flatten() - forward_operator_mat @ recon.flatten())
        jac_sig = recon.flatten() * (1 - recon.flatten())
        hess_sig = (1 - 2*recon.flatten()) * jac_sig
        noise_mat = (1 / lik_variance) * ( torch.diag(hess_sig * loss_grad) + torch.diag(jac_sig) @ A_t_A @ torch.diag(jac_sig) )

    elif mode == 'sigmoid_autograd':
        def func(x):
            return ( 1 / lik_variance) * F.mse_loss(obs.flatten(), forward_operator_mat @ torch.sigmoid(x), reduction='sum')
        noise_mat = 0.5 * AF.hessian(func, recon_no_sigmoid.flatten(), create_graph=False, strict=False, vectorize=False)
    else:
        raise NotImplementedError

    sign, noise_mat_det = torch.linalg.slogdet(noise_mat)
    cnt = 0
    while sign < 0:
        noise_mat[np.diag_indices(noise_mat.shape[0])] += 1e-6
        sign, noise_mat_det = torch.linalg.slogdet(noise_mat)
        cnt += 1
        assert cnt < 100

    U, S, V = tl.truncated_svd(noise_mat, n_eigenvecs=100)
    noise_mat_inv = V.t() @ torch.diag_embed(1 / S) @ U.t() + 0.1 * torch.eye(784)

    return noise_mat, noise_mat_inv, noise_mat_det

def hess_log_det(
    block_priors,
    Jac,
    noise_mat_det,
    noise_mat_inv
    ):

    log_prior_det_inv = -block_priors.get_net_log_det_cov_mat()
    jac_cov_term = block_priors.matrix_prior_cov_mul(Jac) @ Jac.transpose(1, 0)
    (sign, second_term) = torch.linalg.slogdet(noise_mat_inv
            + jac_cov_term)
    cnt = 0
    while sign < 0:
        noise_mat_inv[np.diag_indices(noise_mat_inv.shape[0])] += 1e-6
        (sign, second_term) = torch.linalg.slogdet(noise_mat_inv
                + jac_cov_term)
        cnt += 1
        assert cnt < 100

    return log_prior_det_inv + noise_mat_det + second_term

def optim_marginal_lik(
    cfg,
    observation,
    recon_proj,
    filtbackproj,
    reconstructor,
    block_priors,
    Jac,
    noise_mat_det,
    noise_mat_inv,
    lin_weights = None
    ):


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'mrglik_opt'

    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)

    writer = SummaryWriter(log_dir=logdir)

    log_lik_log_variance = torch.nn.Parameter(torch.zeros(1, device=reconstructor.device))

    reconstructor.model.eval()
    optimizer = \
        torch.optim.Adam([{'params': block_priors.log_lengthscales},
                         {'params': block_priors.log_variances},
                         {'params': log_lik_log_variance}],
                        **{'lr': cfg.mrglik.optim.lr})

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik. opt') as pbar:
        for i in pbar:

            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predCP:
                marginal_lik_PredCP_update(cfg, reconstructor, filtbackproj,
                                           block_priors)

            noise_mat_inv_scaled = torch.exp(log_lik_log_variance) * noise_mat_inv.to(reconstructor.device)
            noise_mat_det_scaled = noise_mat_inv.shape[0] * (-log_lik_log_variance) + noise_mat_det.to(reconstructor.device)


            approx_hess_log_det = hess_log_det(block_priors, Jac,
                    noise_mat_det_scaled,
                    noise_mat_inv_scaled)

            model_log_prob = gaussian_log_prob(observation.to(reconstructor.device), recon_proj.to(reconstructor.device), torch.exp(log_lik_log_variance))

            if cfg.linearize_weights and lin_weights is not None:
                loss = -( model_log_prob + block_priors.get_net_prior_log_prob_lin_weights(lin_weights) - 0.5
                         * approx_hess_log_det)
            else:
                loss = -( model_log_prob + block_priors.get_net_prior_log_prob() - 0.5
                         * approx_hess_log_det)

            loss.backward()
            optimizer.step()

            for k in range(block_priors.num_params):
                writer.add_scalar('lengthscale_{}'.format(k), torch.exp(block_priors.log_lengthscales[k]).item(), i)
                writer.add_scalar('variance_{}'.format(k), torch.exp(block_priors.log_variances[k]).item(), i)
            writer.add_scalar('log_lik_variance', torch.exp(log_lik_log_variance).item(), i)

    return torch.exp(log_lik_log_variance).detach().cpu()
