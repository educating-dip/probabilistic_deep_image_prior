import os
import numpy as np 
import bios
from math import ceil
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
import scipy
import hydra
import torch
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor, PSNR, SSIM
from omegaconf import OmegaConf
from dataset.walnut import get_walnut_data, INNER_PART_START_0, INNER_PART_END_0, INNER_PART_START_1, INNER_PART_END_1, get_inner_block_indices
from dataset.walnuts_interface import get_single_slice_ray_trafo, get_projection_data
from dataset.utils import get_standard_ray_trafos
from scalable_linearised_laplace import get_image_block_mask_inds

DIR_PATH='/localdata/jleuschn/experiments/dip_bayesian_ext/'

# run_path_mll = 'outputs/2022-01-18T22:28:49.118309Z'
# # run_path_mll = 'outputs/2022-01-20T11:29:58.458635Z'  # no sigma_y override, Kyy eps abs 0.1

# # run_path_map = 'outputs/2022-01-18T22:32:09.051642Z'  # Kyy eps rel 1e-5
# run_path_map = 'outputs/2022-01-18T22:34:13.511015Z'  # Kyy eps abs 0.15661916026566242
# # run_path_map = 'outputs/2022-01-20T11:34:15.385454Z'  # no sigma_y override, Kyy eps abs 0.1

# run_path_mcdo = 'outputs/2022-01-18T19:06:20.999879Z'



# only diagonal (1x1 blocks)
# BLOCK_SIZE = 1
# run_path_mll = 'outputs/2022-01-26T00:35:48.232522Z'
# run_path_map = 'outputs/2022-01-25T23:01:38.312512Z'  # tv 50
# # run_path_map = 'outputs/2022-01-26T12:15:47.785793Z'  # tv 5
# run_path_mcdo = 'outputs/2022-01-26T12:48:23.773875Z'

# # 2x2 blocks
BLOCK_SIZE = 2
run_path_mll = 'outputs/2022-01-26T00:55:03.152392Z'
run_path_map = 'outputs/2022-01-26T00:52:20.554709Z'  # tv 50
# run_path_map = 'outputs/2022-01-26T12:16:43.417979Z'  # tv 5
run_path_mcdo = 'outputs/2022-01-26T13:36:28.182717Z'

# # 4x4 blocks
# BLOCK_SIZE = 4
# run_path_mll = 'outputs/2022-01-26T12:07:51.728686Z'
# run_path_map = 'outputs/2022-01-26T12:04:20.319776Z'  # tv 50
# # run_path_map = 'outputs/2022-01-26T12:16:58.319795Z'  # tv 5
# run_path_mcdo = 'outputs/2022-01-26T13:25:49.279219Z'

# # 8x8 blocks
# BLOCK_SIZE = 8
# run_path_mll = 'outputs/2022-01-26T12:08:01.054439Z'
# run_path_map = 'outputs/2022-01-26T12:05:16.778534Z'  # tv 50
# # run_path_map = 'outputs/2022-01-26T12:17:32.412439Z'  # tv 5
# run_path_mcdo = 'outputs/2022-01-26T12:49:52.534479Z'


name = 'walnut'

IM_SHAPE = (501, 501)

START_0 = INNER_PART_START_0
START_1 = INNER_PART_START_1
END_0 = INNER_PART_END_0
END_1 = INNER_PART_END_1

def get_noise_correction_term(ray_trafos, log_noise_model_variance_obs):
    # pseudo-inverse computation
    trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
    U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
    # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
    S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
    # trafo_T_trafo_inv_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
    trafo_T_trafo_inv_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
    lik_hess_inv_diag_mean = np.mean(trafo_T_trafo_inv_diag) * np.exp(log_noise_model_variance_obs)
    return lik_hess_inv_diag_mean

def collect_log_noise_model_variance_obs(path):
    optim_cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    optim_path = optim_cfg.density.compute_single_predictive_cov_block.load_path
    optim_load_iter = optim_cfg.density.compute_single_predictive_cov_block.get('load_mrglik_opt_iter', None)
    log_noise_model_variance_obs = torch.load(os.path.join(optim_path, 'log_noise_model_variance_obs_0.pt' if optim_load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_0_iter_{}.pt'.format(optim_load_iter)))['log_noise_model_variance_obs'].item()
    print(np.exp(log_noise_model_variance_obs))
    return log_noise_model_variance_obs

def collect_reconstruction_data(path, cfg, ray_trafos):

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    observation, filtbackproj, image = get_walnut_data(cfg)

    data_path_test = os.path.join(
            get_original_cwd(), cfg.data_path_test)
    data_path_ray_trafo = os.path.join(
            get_original_cwd(), cfg.ray_trafo_custom.data_path)
    observation_full = get_projection_data(
            data_path=data_path_test,
            walnut_id=cfg.walnut_id, orbit_id=cfg.orbit_id,
            angular_sub_sampling=cfg.angular_sub_sampling,
            proj_col_sub_sampling=cfg.proj_col_sub_sampling)
    # WalnutRayTrafo instance needed for selecting and masking the projections
    walnut_ray_trafo = get_single_slice_ray_trafo(
            data_path_ray_trafo,
            walnut_id=cfg.ray_trafo_custom.walnut_id,
            orbit_id=cfg.ray_trafo_custom.orbit_id,
            angular_sub_sampling=cfg.angular_sub_sampling,
            proj_col_sub_sampling=cfg.proj_col_sub_sampling)
    observation_2d = np.take_along_axis(walnut_ray_trafo.projs_from_full(observation_full), walnut_ray_trafo.proj_mask_first_row_inds[None], axis=0).squeeze()

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

    model_path = os.path.join(
        get_original_cwd(),
        reconstructor.cfg.finetuned_params_path 
        if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
    reconstructor.model.load_state_dict(torch.load(model_path, map_location=reconstructor.device))
    with torch.no_grad():
        reconstructor.model.eval()
        recon, _ = reconstructor.model.forward(torch.from_numpy(filtbackproj)[None, None].to(reconstructor.device))
    recon = recon[0, 0].cpu().numpy()

    abs_error = np.abs(image - recon)

    return (image, observation_2d, filtbackproj, recon, abs_error)

def collect_mcdo_data(path, cfg, block_idx_list, all_block_mask_inds, noise_correction_term=0.):

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    _, _, image = get_walnut_data(cfg)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

    predictive_image_log_prob_dict = torch.load(os.path.join(path, 'predictive_image_log_prob_0.pt'))
    recon_mcdo = predictive_image_log_prob_dict['recon'][0, 0].cpu().numpy()

    abs_error_mcdo = np.abs(image - recon_mcdo)

    std_mcdo = np.full(np.prod(IM_SHAPE), np.nan)

    for mask_inds, block_diag in zip(predictive_image_log_prob_dict['block_mask_inds'], predictive_image_log_prob_dict['block_diags']):
        std_mcdo[mask_inds] = np.clip(block_diag.cpu().numpy() - noise_correction_term, a_min=0., a_max=None)**0.5

    requested_block_mask_inds = np.array(all_block_mask_inds)[block_idx_list]
    # loaded data should be exactly what is requested with block_idx_list
    assert np.array_equal(sorted(np.concatenate(predictive_image_log_prob_dict['block_mask_inds'])),
                          sorted(np.concatenate(requested_block_mask_inds)))
    num_pixels = len(np.concatenate(predictive_image_log_prob_dict['block_mask_inds']))
    log_lik = predictive_image_log_prob_dict['approx_log_prob'].item() / num_pixels

    return (log_lik, recon_mcdo, abs_error_mcdo, std_mcdo.reshape(IM_SHAPE))

def collect_dip_bayes_data(path, cfg, block_idx_list, all_block_mask_inds, noise_correction_term=0.):

    std = np.full(np.prod(IM_SHAPE), np.nan)

    try:
        predictive_image_log_prob_dict = torch.load(os.path.join(DIR_PATH, path, 'predictive_image_log_prob_0.pt'))
        assert np.array_equal(np.concatenate(predictive_image_log_prob_dict['block_mask_inds']), np.concatenate(np.array(all_block_mask_inds)[block_idx_list]))
        for mask_inds, block_diag in zip(predictive_image_log_prob_dict['block_mask_inds'], predictive_image_log_prob_dict['block_diags']):
            std[mask_inds] = np.clip(block_diag.cpu().numpy() - noise_correction_term, a_min=0., a_max=None)**0.5
        block_log_probs = [block_log_prob.item() for block_log_prob in predictive_image_log_prob_dict['block_log_probs']]
        num_pixels = len(np.concatenate(predictive_image_log_prob_dict['block_mask_inds']))
    except FileNotFoundError:
        block_log_probs = []
        num_pixels = 0
        for block_idx, mask_inds in enumerate(all_block_mask_inds):
            if block_idx not in block_idx_list:
                continue
            predictive_image_log_prob_block_dict = torch.load(os.path.join(DIR_PATH, path, 'predictive_image_log_prob_block{}_0.pt'.format(block_idx)))
            assert np.array_equal(mask_inds, predictive_image_log_prob_block_dict['mask_inds'])
            std[mask_inds] = np.clip(predictive_image_log_prob_block_dict['block_diag'].cpu().numpy() - noise_correction_term, a_min=0., a_max=None)**0.5
            block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'].item())
            num_pixels += len(mask_inds)

    log_lik = np.sum(block_log_probs) / num_pixels

    return log_lik, std.reshape(IM_SHAPE)

@hydra.main(config_path='../../cfgs', config_name='config')  # to enable get_original_cwd
def table_walnut(cfg):
# def table_walnut():

    full_run_path_mll = os.path.join(DIR_PATH, run_path_mll)
    full_run_path_map = os.path.join(DIR_PATH, run_path_map)
    full_run_path_mcdo = os.path.join(DIR_PATH, run_path_mcdo)

    cfg = OmegaConf.load(os.path.join(full_run_path_mll, '.hydra', 'config.yaml'))  # use mll config, the settings relevant in this script should be same for all methods
    assert cfg.density.block_size_for_approx == BLOCK_SIZE

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
    image, observation_2d, filtbackproj, recon, abs_error = collect_reconstruction_data(full_run_path_mll, cfg, ray_trafos)
    noise_correction_term_mll = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=collect_log_noise_model_variance_obs(full_run_path_mll))
    noise_correction_term_map = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=collect_log_noise_model_variance_obs(full_run_path_map))
    noise_correction_term_mcdo = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=0.)
    inner_block_idx_list = get_inner_block_indices(BLOCK_SIZE)
    all_block_mask_inds = get_image_block_mask_inds(IM_SHAPE, cfg.density.block_size_for_approx)
    inner_block_mask_inds = np.array(all_block_mask_inds)[inner_block_idx_list]
    inner_inds = np.concatenate(inner_block_mask_inds)

    log_lik_mll, _ = collect_dip_bayes_data(full_run_path_mll, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_mll)
    log_lik_map, _ = collect_dip_bayes_data(full_run_path_map, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_map)
    log_lik_mcdo, recon_mcdo, _, _ = collect_mcdo_data(full_run_path_mcdo, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_mcdo)

    inner_image = image.reshape(-1)[inner_inds]
    inner_recon = recon.reshape(-1)[inner_inds]
    inner_recon_mcdo = recon_mcdo.reshape(-1)[inner_inds]

    include_metrics = True

    table = ''
    table += ' & log-likelihood'
    if include_metrics:
        table += ' & PSNR [dB] & SSIM'
    table += '\\\\\n'

    table += 'Bayes DIP (MLL) & ${:.4f}$'.format(log_lik_mll)
    if include_metrics:
        table += ' & ${:.3f}$'.format(PSNR(recon[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
        table += ' & ${:.4f}$'.format(SSIM(recon[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
    table += '\\\\\n'
    table += 'Bayes DIP (TV-MAP) & ${:.4f}$'.format(log_lik_map)
    if include_metrics:
        table += ' & ${:.3f}$'.format(PSNR(recon[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
        table += ' & ${:.4f}$'.format(SSIM(recon[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
    table += '\\\\\n'
    table += 'DIP-MCDO & ${:.4f}$'.format(log_lik_mcdo)
    if include_metrics:
        table += ' & ${:.3f}$'.format(PSNR(recon_mcdo[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
        table += ' & ${:.4f}$'.format(SSIM(recon_mcdo[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1]))
    table += '\\\\\n'

    print(table)

    with open(f'table_walnut_{BLOCK_SIZE}x{BLOCK_SIZE}.txt', 'w') as f:
        f.write(table)

if __name__ == "__main__": 
    table_walnut()
