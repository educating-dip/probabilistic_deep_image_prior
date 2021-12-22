from .batch_jac import vec_jac_mul_batch
from .jvp import fwAD_JvP_batch_ensemble, finite_diff_JvP_batch_ensemble
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul
import torch
import numpy as np


# reference for testing
def agregate_flatten_weight_grad(modules):
    grads = []
    for layer in modules:
        assert isinstance(layer, torch.nn.Conv2d)
        grads.append(layer.weight.grad.flatten())
    return torch.cat(grads)

def vec_jac_mul_single(model, modules, filtbackproj, v):

    jac = []
    model.eval()
    f = model(filtbackproj)[0]
    model.zero_grad()
    f.backward(v, retain_graph=True)
    v_jac = agregate_flatten_weight_grad(modules).detach()
    return v_jac


# multiply v with Kyy and add sigma_y * v
def prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, v, log_noise_model_variance_obs, use_fwAD_for_jvp=True):
    v_image = ray_trafos['ray_trafo_module_adj'](v)
    image_shape = v_image.shape

    # v_image = v_image.view(v_image.shape[0], -1)
    # v_params = vec_jac_mul_batch(hooked_model, filtbackproj, v_image)
    v_params_list = [vec_jac_mul_single(hooked_model, bayesianized_model.get_all_modules_under_prior(), filtbackproj, v_image_i[None]) for v_image_i in v_image]
    v_params = torch.stack(v_params_list)

    v_params = vec_weight_prior_cov_mul(bayesianized_model, v_params)
    if use_fwAD_for_jvp:
        v_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, v_params, be_modules)
    else:
        v_image = finite_diff_JvP_batch_ensemble(filtbackproj, be_model, v_params, be_modules)
    v_image = torch.squeeze(v_image, dim=1)  # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
    v_obs = ray_trafos['ray_trafo_module'](v_image)
    v_obs = v_obs + v * torch.exp(log_noise_model_variance_obs)  # TODO check that no square is needed
    return v_obs

# build Kyy
def get_prior_cov_obs_mat(ray_trafos, filtbackproj, bayesianized_model, hooked_model, be_model, be_modules, log_noise_model_variance_obs, use_fwAD_for_jvp=True, vec_batch_size=1):
    obs_shape = ray_trafos['ray_trafo'].range.shape
    rows = []
    v = torch.zeros((vec_batch_size,) + obs_shape)
    # full batches
    for i in range(0, np.prod(obs_shape), vec_batch_size):
        # TODO set zero
        v.view(vec_batch_size, -1)[i:i+vec_batch_size, :].fill_diagonal_(1.)
        row = ...
        rows.append(row)
    # last batch
    # TODO
    cov_obs_mat = torch.cat(rows, dim=0)
    return cov_obs_mat
