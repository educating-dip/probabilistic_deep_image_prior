import torch
import numpy as np
from tqdm import tqdm
from .density import get_cov_image_mat
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul
from .batch_jac import vec_jac_mul_batch
from .jvp import fwAD_JvP_batch_ensemble
from .prior_cov_obs import get_prior_cov_obs_mat

def predictive_cov_image_block_norm(v, cov_image_mat_block):
    v_out = torch.linalg.solve(cov_image_mat_block, v.T).T
    return v.T @ v_out

def predictive_image_block_log_prob(recon_masked, ground_truth_masked, predictive_cov_image_block):
    approx_slogdet = torch.slogdet(predictive_cov_image_block)
    assert approx_slogdet[0] > 0.
    approx_log_det = approx_slogdet[1]
    diff = (ground_truth_masked - recon_masked).flatten()
    norm = predictive_cov_image_block_norm(diff, predictive_cov_image_block)
    approx_log_prob = -0.5 * norm - 0.5 * approx_log_det + -0.5 * np.log(2. * np.pi) * ground_truth_masked.numel()
    return approx_log_prob

def cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules):
    v = vec_jac_mul_batch(hooked_model, filtbackproj, v, bayesianized_model)  # v * J
    v = vec_weight_prior_cov_mul(bayesianized_model, v)  # v * Σ_θ
    v = fwAD_JvP_batch_ensemble(filtbackproj, fwAD_be_model, v, fwAD_be_modules)  # v * J.T
    v = v.view(v.shape[0], -1)
    return v

# v @ K_f|y
def predictive_cov_image_block_mul(v, mask, cov_image_mat_block, cov_obs_mat_chol, ray_trafos, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules):
    v_input = v
    v = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
    v = ray_trafos['ray_trafo_module'](v.view(v.shape[0], 1, *ray_trafos['space'].shape)).view(v.shape[0], -1)
    v = torch.triangular_solve(torch.triangular_solve(v.T, cov_obs_mat_chol, upper=False)[0], cov_obs_mat_chol.T, upper=True)[0].T
    v = ray_trafos['ray_trafo_module_adj'](v.view(v.shape[0], 1, *ray_trafos['ray_trafo'].range.shape)).view(v.shape[0], -1)
    v = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
    # v = v_input @ K_ff @ A.T @ Kyy^-1 @ A @ K_ff
    v = v_input[:, mask] @ cov_image_mat_block - v[:, mask]
    return v

def get_image_block_slices(image_shape, block_size):
    image_size_0, image_size_1 = image_shape
    block_size = min(block_size, min(*image_shape))

    block_slices_0 = []
    for start_0 in range(0, image_size_0 - (block_size-1), block_size):
        if start_0 + block_size < image_size_0 - (block_size-1):
            end_0 = start_0 + block_size
        else:
            # last full block, also include the remaining pixels
            end_0 = image_size_0
        block_slices_0.append(slice(start_0, end_0))
    block_slices_1 = []
    for start_1 in range(0, image_size_1 - (block_size-1), block_size):
        if start_1 + block_size < image_size_1 - (block_size-1):
            end_1 = start_1 + block_size
        else:
            # last full block, also include the remaining pixels
            end_1 = image_size_1
        block_slices_1.append(slice(start_1, end_1))
    return block_slices_0, block_slices_1

def get_image_block_masks(image_shape, block_size, flatten=True):
    block_slices_0, block_slices_1 = get_image_block_slices(image_shape, block_size)

    block_masks = []
    for slice_0 in block_slices_0:
        for slice_1 in block_slices_1:
            mask = np.zeros(image_shape, dtype=bool)
            mask[slice_0, slice_1] = True
            if flatten:
                mask = mask.flatten()
            block_masks.append(mask)
    return block_masks

# block of K_ff
def get_cov_image_mat_block(mask, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=None):

    image_shape = (1, 1,) + ray_trafos['space'].shape
    mask_inds = np.nonzero(mask)[0]
    block_numel = len(mask_inds)
    rows = []
    v = torch.empty((vec_batch_size, np.prod(image_shape)), device=filtbackproj.device)
    for i in range(0, block_numel, vec_batch_size):
        v[:] = 0.
        # set v[:, mask] to be a subset of rows of torch.eye(block_numel); in last batch, it may contain some additional (zero) rows
        mask_inds_batch = mask_inds[i:i+vec_batch_size]
        v[list(range(len(mask_inds_batch))), mask_inds_batch] = 1.
        rows_batch = cov_image_mul(v, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
        rows_batch = rows_batch[:, mask]
        if i+vec_batch_size > block_numel:  # last batch
            rows_batch = rows_batch[:block_numel%vec_batch_size]
        rows.append(rows_batch)
    cov_image_mat_block = torch.cat(rows, dim=0)
    if eps is not None:
        cov_image_mat_block[np.diag_indices(cov_image_mat_block.shape[0])] += eps
    return cov_image_mat_block

# block of K_f|y
def get_predictive_cov_image_block(mask, cov_obs_mat_chol, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=None, cov_image_eps=None, return_cholesky=False):
    cov_image_mat_block = get_cov_image_mat_block(mask, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=cov_image_eps)

    image_shape = (1, 1,) + ray_trafos['space'].shape
    mask_inds = np.nonzero(mask)[0]
    block_numel = len(mask_inds)
    rows = []
    v = torch.empty((vec_batch_size, np.prod(image_shape)), device=filtbackproj.device)
    for i in range(0, block_numel, vec_batch_size):
        v[:] = 0.
        # set v[:, mask] to be a subset of rows of torch.eye(block_numel); in last batch, it may contain some additional (zero) rows
        mask_inds_batch = mask_inds[i:i+vec_batch_size]
        v[list(range(len(mask_inds_batch))), mask_inds_batch] = 1.
        rows_batch = predictive_cov_image_block_mul(v, mask, cov_image_mat_block, cov_obs_mat_chol, ray_trafos, filtbackproj, hooked_model, bayesianized_model, fwAD_be_model, fwAD_be_modules)
        if i+vec_batch_size > block_numel:  # last batch
            rows_batch = rows_batch[:block_numel%vec_batch_size]
        rows.append(rows_batch)
    predictive_cov_image_block = torch.cat(rows, dim=0)
    if eps is not None:
        predictive_cov_image_block[np.diag_indices(predictive_cov_image_block.shape[0])] += eps

    assert torch.all(predictive_cov_image_block.diag() > 0.)

    return torch.linalg.cholesky(predictive_cov_image_block) if return_cholesky else predictive_cov_image_block

def predictive_image_log_prob(
        recon, ground_truth, ray_trafos, bayesianized_model, filtbackproj, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, eps, cov_image_eps, block_size, vec_batch_size):

    device = filtbackproj.device

    block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size, flatten=True)

    cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True)
    cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)

    image_block_diags = []
    image_block_log_probs = []
    for mask in tqdm(block_masks, desc='image_block_log_probs'):
        predictive_cov_image_block = get_predictive_cov_image_block(
                mask, cov_obs_mat_chol, ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, vec_batch_size, eps=eps, cov_image_eps=cov_image_eps, return_cholesky=False)

        image_block_diags.append(predictive_cov_image_block.diag())
        image_block_log_probs.append(predictive_image_block_log_prob(recon.flatten()[mask], ground_truth.flatten()[mask], predictive_cov_image_block))

    approx_image_log_prob = torch.sum(torch.stack(image_block_log_probs))

    return approx_image_log_prob, block_masks, image_block_log_probs, image_block_diags
