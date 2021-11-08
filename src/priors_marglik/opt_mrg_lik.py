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
from linearized_laplace import compute_submatrix_lin_pred_cov
import tensorly as tl
tl.set_backend('pytorch')

def marginal_lik_PredCP_linear_update(
    cfg, 
    block_priors, 
    Jac, 
    recon_no_sigmoid
    ):


    _, list_prior_cov_pred = compute_submatrix_lin_pred_cov(block_priors, Jac)
    expected_tv = []
    for cov in list_prior_cov_pred:
        succed = False
        cnt = 0
        while not succed: 
            try: 
                dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=recon_no_sigmoid.flatten().cuda(), covariance_matrix=cov)
                succed = True 
            except: 
                cov[np.diag_indices(cov.shape[0])] += 1e-2
                cnt += 1
            assert cnt < 100

        samples = dist.rsample((100, ))
        expected_tv.append(tv_loss(torch.sigmoid(samples)).mean(dim=0))

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
    
    loss =  cfg.mrglik.optim.scaling_fct * cfg.mrglik.optim.scl_fct_gamma * cfg.mrglik.optim.gamma * torch.stack(expected_tv).sum().detach() + torch.stack(log_det_list).sum().detach()
    return loss

def marginal_lik_PredCP_update(
    cfg,
    filtbackproj,
    block_priors
    ):

    expected_tv = block_priors.get_expected_TV_loss(filtbackproj)
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
    
    loss =  cfg.mrglik.optim.scaling_fct * cfg.mrglik.optim.scl_fct_gamma * cfg.mrglik.optim.gamma * torch.stack(expected_tv).sum().detach() + torch.stack(log_det_list).sum().detach()
    return loss

def compute_lik_hess(
    obs, 
    recon, 
    recon_no_sigmoid, 
    trafo, 
    trafo_adj,
    trafo_adj_trafo,
    mode='sigmoid_exact'):

    if mode == 'linear':
        lik_hess = trafo_adj_trafo

    elif mode == 'sigmoid_exact':
        loss_grad = - trafo_adj @ (obs.flatten() - trafo @ recon.flatten())
        jac_sig = recon.flatten() * (1 - recon.flatten())
        hess_sig = (1 - 2*recon.flatten()) * jac_sig
        lik_hess = torch.diag(hess_sig * loss_grad) + torch.diag(jac_sig) @ trafo_adj_trafo @ torch.diag(jac_sig)

    elif mode == 'sigmoid_autograd':
        def func(x):
            return F.mse_loss(obs.flatten(), trafo @ torch.sigmoid(x), reduction='sum')
        lik_hess = 0.5 * AF.hessian(func, recon_no_sigmoid.flatten(), create_graph=False, strict=False, vectorize=False)
    else:
        raise NotImplementedError

    U, S, Vh = tl.truncated_svd(lik_hess, n_eigenvecs=50)

    return lik_hess, (U, S, Vh)

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
    recon, 
    recon_no_sigmoid, 
    filtbackproj,
    block_priors,
    Jac,
    ray_trafos, 
    lin_weights = None
    ):


    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'mrglik_opt'

    logdir = os.path.join(
        './',
        current_time + '_' + socket.gethostname() + comment)

    writer = SummaryWriter(log_dir=logdir)

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    log_lik_log_variance_y = torch.nn.Parameter(torch.zeros(1,
            device=device))
    log_lik_log_variance_x = torch.nn.Parameter(torch.zeros(1,
            device=device))

    trafo, trafo_adj, trafo_adj_trafo, trafo_trafo_adj = extract_tafo_as_matrix(ray_trafos)

    _, (U, S, Vh) = compute_lik_hess(
        observation,
        recon,
        recon_no_sigmoid,
        trafo,
        trafo_adj,
        trafo_adj_trafo,
        mode='sigmoid_exact',
        )

    U, S, Vh = U.to(device), S.to(device), Vh.to(device)

    recon_proj = (trafo @ recon.flatten()).to(device)
    observation = observation.to(device).flatten()

    optimizer = \
        torch.optim.Adam([{'params': block_priors.log_lengthscales, 'lr': cfg.mrglik.optim.lr},
                         {'params': block_priors.log_variances, 'lr': cfg.mrglik.optim.lr},
                         {'params': log_lik_log_variance_y, 'lr': cfg.mrglik.optim.lr},
                         {'params': log_lik_log_variance_x, 'lr': cfg.mrglik.optim.lr}],
                        )

    with tqdm(range(cfg.mrglik.optim.iterations), desc='mrglik. opt') as pbar:
        for i in pbar:

            optimizer.zero_grad()
            if cfg.mrglik.optim.include_predCP:
                # predCP_loss = marginal_lik_PredCP_update(cfg, filtbackproj, block_priors)
                predCP_loss = marginal_lik_PredCP_linear_update(cfg, block_priors, Jac, recon_no_sigmoid)

            else: 
                predCP_loss = torch.zeros(1)

            log_lik_variance_x = torch.exp(log_lik_log_variance_x) + 1e-1

            lik_hess_inv = Vh.t() @ torch.diag_embed( torch.exp(log_lik_log_variance_y) / S) @ U.t() \
                        + log_lik_variance_x * torch.eye(784, device=device)

            lik_hess_inv_det = torch.log(torch.cat((torch.exp(log_lik_log_variance_y) / S,
                        torch.zeros(784-S.shape[0], device=device))) + log_lik_variance_x).sum()

            assert lik_hess_inv.diag().min() > 0

            approx_hess_log_det = hess_log_det(block_priors, Jac,
                    lik_hess_inv_det,
                    lik_hess_inv)

            log_lik_model_variance = torch.exp(log_lik_log_variance_y) \
                + log_lik_variance_x \
                * trafo_trafo_adj.diag().to(device) + 1e-6

            assert log_lik_model_variance.min() > 0 

            model_log_prob = gaussian_log_prob(observation, recon_proj, log_lik_model_variance)

            if cfg.linearize_weights and lin_weights is not None:
                prior_log_prob = block_priors.get_net_prior_log_prob_lin_weights(lin_weights)
            else:
                prior_log_prob = block_priors.get_net_prior_log_prob()
            
            loss = -(model_log_prob + prior_log_prob - 0.5 * approx_hess_log_det)

            loss.backward()
            optimizer.step()

            for k in range(block_priors.num_params):
                writer.add_scalar('lengthscale_{}'.format(k), torch.exp(block_priors.log_lengthscales[k]).item(), i)
                writer.add_scalar('variance_{}'.format(k), torch.exp(block_priors.log_variances[k]).item(), i)

            writer.add_scalar('negative MLL', loss.item() - predCP_loss.item(), i)
            writer.add_scalar('model log prob', model_log_prob.item(), i)
            writer.add_scalar('post_hess_log_det', approx_hess_log_det.item(), i)
            writer.add_scalar('prior_log_prob', prior_log_prob.item(), i)
            writer.add_scalar('predCP', predCP_loss.item(), i)
            writer.add_scalar('log_lik_variance_y', torch.exp(log_lik_log_variance_y).item(), i)
            writer.add_scalar('log_lik_variance_x', log_lik_variance_x.item(), i)

    return torch.exp(log_lik_log_variance_y).detach().cpu(), torch.exp(log_lik_log_variance_x).detach().cpu(), lik_hess_inv.detach()
