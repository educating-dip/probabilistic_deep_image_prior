import torch
from math import ceil 
from sklearn.neighbors import KernelDensity
from .jvp import fwAD_JvP_batch_ensemble
from .mc_pred_cp_loss import _sample_from_prior_over_weights
from .approx_density import cov_image_mul

def sample_from_posterior(ray_trafos, observation, filtbackproj, cov_obs_mat, hooked_model, bayesianized_model, be_model, be_modules, mc_samples, vec_batch_size):
    
    num_batches = ceil(mc_samples / vec_batch_size)
    mc_samples = num_batches * vec_batch_size
    s_images = []
    for _ in range(num_batches):
        sample_weight_vec = _sample_from_prior_over_weights(bayesianized_model, vec_batch_size)
        s_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec, be_modules)
        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        s_observation = ray_trafos['ray_trafo_module'](s_image)

        obs_diff = (observation.expand(*s_observation.shape) - s_observation).view(vec_batch_size, -1)
        inv_diff = ray_trafos['ray_trafo_module_adj']((torch.linalg.solve(cov_obs_mat, obs_diff.T).T).view(vec_batch_size, *observation.shape[1:]))
        delta_x = cov_image_mul(inv_diff.view(vec_batch_size, -1), filtbackproj, hooked_model, bayesianized_model, be_model, be_modules)

        s_images.append(s_image + delta_x.view(vec_batch_size, *s_image.shape[1:]))
    s_images = torch.cat(s_images, axis=0)

    return s_images


def approx_density_from_samples(recon, example_image, mc_sample_images, noise_x_correction_term=None):

    assert example_image.shape[1:] == mc_sample_images.shape[1:]
    
    mc_samples = mc_sample_images.shape[0]
    assert noise_x_correction_term is not None

    std = ( torch.var(mc_sample_images.view(mc_samples, -1), dim=0) + noise_x_correction_term) **.5
    dist = torch.distributions.normal.Normal(recon.flatten(), std)
    return dist.log_prob(example_image.flatten()).sum()


def approx_kernel_density(example_image, mc_sample_images, bw=0.1, noise_x_correction_term=None):

    if noise_x_correction_term is not None:
        mc_sample_images = mc_sample_images + torch.randn_like(mc_sample_images) * noise_x_correction_term **.5

    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(mc_sample_images.view(mc_sample_images.shape[0], -1).numpy())
    return kde.score_samples(example_image.flatten().numpy()[None, :])