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

def marginal_lik_PredCP_update(
    cfg,
    reconstructor,
    filtbackproj,
    block_priors,
    store_device
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

        block_priors.log_lengthscales[i].grad = -(-first_derivative * 2
                * cfg.net.optim.gamma * cfg.mrglik.optim.scl_fct_gamma + second_derivative)
        block_priors.log_variances[i].grad = \
            -first_derivative_log_variances * 2 * cfg.net.optim.gamma * cfg.mrglik.optim.scl_fct_gamma \
            + second_derivative_log_variances

def estimate_lik_noise(obs, filtbackproj, reconstructor, ray_trafos, mode='sigmoid_exact'):

    forward_operator_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
    forward_operator_mat = forward_operator_mat.view(-1, ray_trafos['space'].shape[0]**2)
    forward_operator_mat_adj = forward_operator_mat.t()

    if mode == 'linear':
        noise_mat = forward_operator_mat_adj @ forward_operator_mat

    elif mode == 'sigmoid_exact':
        recon = reconstructor.model.forward(filtbackproj)[0].detach().cpu().flatten()
        loss_grad = - forward_operator_mat_adj @ (obs.flatten() - forward_operator_mat @ recon)
        jac_sig = recon * (1 - recon)
        hess_sig = (1 - 2*recon) * jac_sig
        noise_mat = torch.diag(hess_sig * loss_grad) + torch.diag(jac_sig) @ forward_operator_mat_adj @ forward_operator_mat @ torch.diag(jac_sig)

    elif mode == 'sigmoid_autograd':
        out_no_sigmoid = reconstructor.model.forward(filtbackproj)[1].detach().cpu().flatten()
        def func(x):
            return F.mse_loss(obs.flatten(), forward_operator_mat @ torch.sigmoid(x), reduction='sum')
        noise_mat = 0.5 * AF.hessian(func, out_no_sigmoid, create_graph=False, strict=False, vectorize=False)
    else:
        raise NotImplementedError

    sign, noise_mat_det = torch.linalg.slogdet(noise_mat)
    cnt = 0
    while sign < 0:
        noise_mat[np.diag_indices(noise_mat.shape[0])] += 1e-3
        sign, noise_mat_det = torch.linalg.slogdet(noise_mat)
        cnt += 1
        assert cnt < 10

    noise_mat_inv = torch.inverse(noise_mat)

    return noise_mat, noise_mat_inv, noise_mat_det

def hess_log_det(
    block_priors,
    Jac,
    noise_mat_det,
    noise_mat_inv,
    eps=1e-6,
    ):

    log_prior_det_inv = -block_priors.get_net_log_det_cov_mat()
    jac_cov_term = block_priors.matrix_prior_cov_mul(Jac) @ Jac.transpose(1, 0)
    (sign, second_term) = torch.linalg.slogdet(noise_mat_inv
            + jac_cov_term)
    cnt = 0
    while sign < 0:
        noise_mat_inv[np.diag_indices(noise_mat_inv.shape[0])] += eps
        (sign, second_term) = torch.linalg.slogdet(noise_mat_inv
                + jac_cov_term)
        cnt += 1
        assert cnt < 10

    return log_prior_det_inv + noise_mat_det + second_term

def optim_marginal_lik(
    cfg,
    reconstructor,
    filtbackproj,
    block_priors,
    Jac,
    noise_mat_det,
    noise_mat_inv,
    store_device,
    ):


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'mrglik_opt'

    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)

    writer = SummaryWriter(log_dir=logdir)
    reconstructor.model.eval()
    optimizer = \
        torch.optim.Adam([{'params': block_priors.log_lengthscales},
                         {'params': block_priors.log_variances}],
                        **{'lr': cfg.mrglik.optim.lr})

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik. opt') as pbar:
        for i in pbar:
            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predCP:
                marginal_lik_PredCP_update(cfg, reconstructor, filtbackproj,
                                           block_priors, store_device,
                                           )
            approx_hess_log_det = hess_log_det(block_priors, Jac,
                    noise_mat_det.to(store_device),
                    noise_mat_inv.to(store_device))
            loss = -(block_priors.get_net_prior_log_prob() - 0.5
                     * approx_hess_log_det)
            loss.backward()
            optimizer.step()

            for k in range(block_priors.num_params):
                writer.add_scalar('lengthscale_{}'.format(k), torch.exp(block_priors.log_lengthscales[k]).item(), i)
                writer.add_scalar('variance_{}'.format(k), torch.exp(block_priors.log_variances[k]).item(), i)
