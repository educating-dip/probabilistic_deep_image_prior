from functools import lru_cache
from .batch_jac import vec_op_jac_mul_batch
from .jvp import fwAD_JvP_batch_ensemble, finite_diff_JvP_batch_ensemble
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul, vec_weight_prior_cov_mul_base
from .utils import bisect_left  # for python >= 3.10 one can use instead: from bisect import bisect_left
import torch
import numpy as np
from tqdm import tqdm

# reference for testing
def agregate_flatten_weight_grad(modules):
    grads = []
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        grads.append(layer.weight.grad.flatten())
    return torch.cat(grads)
# reference for testing
def vec_jac_mul_single(model, modules, filtbackproj, v):

    model.eval()
    f = model(filtbackproj)[0]
    model.zero_grad()
    f.backward(v, retain_graph=True)
    v_jac = agregate_flatten_weight_grad(modules).detach()
    return v_jac

# multiply v with Kyy and add σ^2_y * v
def prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, v, log_noise_model_variance_obs, masked_cov_grads=None, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True):
    
    if len(v.shape) == 5:
        v = torch.squeeze(v, dim=1)
    assert len(v.shape) == 4

    # computing v_θ = v * A * J
    v_params = vec_op_jac_mul_batch(ray_trafos, hooked_model, filtbackproj, v, bayesianized_model)

    # computing v_θ = v_θ * Σ_θ
    if masked_cov_grads is None:
        with torch.no_grad():
            v_params = vec_weight_prior_cov_mul(bayesianized_model, v_params)
    # computing v_θ = v_θ * (δΣ_θ / δ_hyperparams)
    else:
        masked_cov_grad_gp_priors, masked_cov_grad_normal_priors = masked_cov_grads
        with torch.no_grad():
            v_params = vec_weight_prior_cov_mul_base(bayesianized_model, masked_cov_grad_gp_priors, masked_cov_grad_normal_priors, v_params)

    # computing v_obs = v_θ * J.T * A.T 
    if use_fwAD_for_jvp:
        v_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, v_params, be_modules)
    else:
        v_image = finite_diff_JvP_batch_ensemble(filtbackproj, be_model, v_params, be_modules)
    v_image = torch.squeeze(v_image, dim=1)  # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
    v_obs = ray_trafos['ray_trafo_module'](v_image)

    # adding σ^2_y * v
    if add_noise_model_variance_obs: 
        v_obs = v_obs + v * torch.exp(log_noise_model_variance_obs)
    return v_obs

# multiply v with diag Kyy and + σ^2_y * v
def prior_diag_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, v, log_noise_model_variance_obs, replace_by_identity=False):
    # diag (Kyy) = e_i A J Σ_θ J.T A.T e_i.T + σ^2_y

    if len(v.shape) == 5:
        v = torch.squeeze(v, dim=1)
    assert len(v.shape) == 4

    # computing v_θ = e_i * A * J
    v = vec_op_jac_mul_batch(ray_trafos, hooked_model, filtbackproj, v, bayesianized_model) # batch_size, num_params

    # computing v_θ = v * Σ_θ
    if not replace_by_identity:
        with torch.no_grad():
            v_params = vec_weight_prior_cov_mul(bayesianized_model, v) # batch_size, num_params
    # Σ_θ = I
    else: 
        v_params = v
    return (v_params * v).sum(dim=1) + torch.exp(log_noise_model_variance_obs) 

# build Kyy
def get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, log_noise_model_variance_obs, vec_batch_size, masked_cov_grads=None, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True, return_on_cpu=False, sub_slice_batches=None):
    obs_shape = (1, 1,) + ray_trafos['ray_trafo'].range.shape
    obs_numel = np.prod(obs_shape)
    if sub_slice_batches is None:
        sub_slice_batches = slice(None)
    rows = []
    v = torch.empty((vec_batch_size,) + obs_shape, device=filtbackproj.device)
    for i in tqdm(np.array(range(0, obs_numel, vec_batch_size))[sub_slice_batches], desc='get_prior_cov_obs_mat', miniters=obs_numel//vec_batch_size//100):
        v[:] = 0.
        # set v.view(vec_batch_size, -1) to be a subset of rows of torch.eye(obs_numel); in last batch, it may contain some additional (zero) rows
        v.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(1.)
        rows_batch = prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, v, log_noise_model_variance_obs, masked_cov_grads=masked_cov_grads, use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=add_noise_model_variance_obs)
        rows_batch = rows_batch.view(vec_batch_size, -1)
        if i+vec_batch_size > obs_numel:  # last batch
            rows_batch = rows_batch[:obs_numel%vec_batch_size]
        rows_batch = rows_batch.cpu()  # collect on CPU (saves memory while running the closure)
        rows.append(rows_batch)
    cov_obs_mat = torch.cat(rows, dim=0)
    return cov_obs_mat if return_on_cpu else cov_obs_mat.to(filtbackproj.device)

# build diag Kyy
def get_diag_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, log_noise_model_variance_obs, vec_batch_size, replace_by_identity=False):
    obs_shape = (1, 1,) + ray_trafos['ray_trafo'].range.shape
    obs_numel = np.prod(obs_shape)
    rows = []
    v = torch.empty((vec_batch_size,) + obs_shape, device=filtbackproj.device)
    for i in tqdm(range(0, obs_numel, vec_batch_size), desc='get_diag_prior_cov_obs_mat', miniters=obs_numel//vec_batch_size//100):
        v[:] = 0.
        # set v.view(vec_batch_size, -1) to be a subset of rows of torch.eye(obs_numel); in last batch, it may contain some additional (zero) rows
        v.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(1.)
        rows_batch = prior_diag_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, v, log_noise_model_variance_obs, replace_by_identity=replace_by_identity)
        rows_batch = rows_batch.view(vec_batch_size, -1)
        if i+vec_batch_size > obs_numel:  # last batch
            rows_batch = rows_batch[:obs_numel%vec_batch_size]
        rows.append(rows_batch.cpu())
    diag_cov_obs_mat = torch.cat(rows, dim=0)
    return diag_cov_obs_mat.squeeze(dim=-1).to(filtbackproj.device)

def stabilize_prior_cov_obs_mat(cov_obs_mat, eps_mode, eps):
    cov_obs_mat_diag_mean = cov_obs_mat.diag().mean().detach().cpu().numpy()
    if eps_mode == 'abs':
        cov_obs_mat_eps = eps or 0.
    elif eps_mode == 'rel':
        cov_obs_mat_eps = (eps or 0.) * cov_obs_mat.diag().mean().detach().cpu().numpy()
    elif eps_mode == 'auto':
        @lru_cache(maxsize=None)
        def cov_obs_mat_cholesky_decomposable(eps_value):
            try:
                _ = torch.linalg.cholesky(cov_obs_mat + eps_value * torch.eye(cov_obs_mat.shape[0], device=cov_obs_mat.device))
            except RuntimeError:
                return False
            return True
        eps_to_search = [0.] + (list(np.logspace(-6, 0, 1000) * eps * cov_obs_mat_diag_mean) if eps else [])
        i_eps = bisect_left(eps_to_search, True, key=cov_obs_mat_cholesky_decomposable)
        assert i_eps < len(eps_to_search), 'failed to make Kyy cholesky decomposable, max eps is {} == {} * Kyy.diag().mean()'.format(eps_to_search[-1], eps_to_search[-1] / cov_obs_mat_diag_mean)
        cov_obs_mat_eps = eps_to_search[i_eps]
    elif eps_mode is None or eps_mode.lower() == 'none':
        cov_obs_mat_eps = 0.
    else:
        raise NotImplementedError
    if cov_obs_mat_eps != 0.:
        cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += cov_obs_mat_eps
        print('increased diagonal of Kyy by {} == {} * Kyy.diag().mean()'.format(cov_obs_mat_eps, cov_obs_mat_eps / cov_obs_mat_diag_mean))

# # build diag Kyy
# def get_diag_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, log_noise_model_variance_obs, vec_batch_size, replace_by_identity=False):
#     obs_shape = (1, 1,) + ray_trafos['ray_trafo'].range.shape
#     obs_numel = np.prod(obs_shape)
#     rows = []
#     v = torch.empty((vec_batch_size,) + obs_shape, device=filtbackproj.device)
#     from tqdm import tqdm
#     for i in tqdm(range(0, obs_numel, vec_batch_size), desc='get_diag_prior_cov_obs_mat'):
#         v[:] = 0.
#         # set v.view(vec_batch_size, -1) to be a subset of rows of torch.eye(obs_numel); in last batch, it may contain some additional (zero) rows
#         v.view(vec_batch_size, -1)[:, i:i+vec_batch_size].fill_diagonal_(1.)

#         # computing v_θ = e_i * A * J
#         # v = vec_op_jac_mul_batch(ray_trafos, hooked_model, filtbackproj, v, bayesianized_model) # batch_size, num_params
#         # v_out = ray_trafos['ray_trafo_module_adj'](torch.squeeze(v, dim=1)).view(v.shape[0], -1)
#         v_out = torch.from_numpy(ray_trafos['ray_trafo_mat_adj'][:, i:i+vec_batch_size].T.todense()).to(filtbackproj.device)  # slicing is supported in scipy but not in torch

#         v_out = vec_jac_mul_batch(hooked_model, filtbackproj, v_out, bayesianized_model) # batch_size, num_params

#         # computing v_θ = v * Σ_θ
#         if not replace_by_identity: 
#             v_params = vec_weight_prior_cov_mul(bayesianized_model, v_out) # batch_size, num_params
#         # Σ_θ = I
#         else: 
#             v_params = v_out
#         rows_batch = (v_params * v_out).sum(dim=1) + torch.exp(log_noise_model_variance_obs)
#         rows_batch = rows_batch.view(vec_batch_size, -1)
#         if i+vec_batch_size > obs_numel:  # last batch
#             rows_batch = rows_batch[:obs_numel%vec_batch_size]
#         rows.append(rows_batch)
#     diag_cov_obs_mat = torch.cat(rows, dim=0)
#     return diag_cov_obs_mat.squeeze(dim=-1)
