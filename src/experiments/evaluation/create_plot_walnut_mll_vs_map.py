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
IMAGES_DIR='./images_walnut'

# run_path_mll = 'outputs/2022-01-18T22:28:49.118309Z'
# # run_path_mll = 'outputs/2022-01-20T11:29:58.458635Z'  # no sigma_y override, Kyy eps abs 0.1

# # run_path_map = 'outputs/2022-01-18T22:32:09.051642Z'  # Kyy eps rel 1e-5
# run_path_map = 'outputs/2022-01-18T22:34:13.511015Z'  # Kyy eps abs 0.15661916026566242
# # run_path_map = 'outputs/2022-01-20T11:34:15.385454Z'  # no sigma_y override, Kyy eps abs 0.1
# # run_path_map = 'outputs/2022-01-24T00:26:29.402003Z'

# run_path_mcdo = 'outputs/2022-01-18T19:06:20.999879Z'


# only diagonal (1x1 blocks)
BLOCK_SIZE = 1
run_path_mll = 'outputs/2022-01-26T00:35:48.232522Z'
run_path_map = 'outputs/2022-01-25T23:01:38.312512Z'  # tv 50
# run_path_map = 'outputs/2022-01-26T12:15:47.785793Z'  # tv 5
run_path_mcdo = 'outputs/2022-01-26T12:48:23.773875Z'

# # 2x2 blocks
# BLOCK_SIZE = 2
# run_path_mll = 'outputs/2022-01-26T00:55:03.152392Z'
# run_path_map = 'outputs/2022-01-26T00:52:20.554709Z'  # tv 50
# # run_path_map = 'outputs/2022-01-26T12:16:43.417979Z'  # tv 5
# run_path_mcdo = 'outputs/2022-01-26T13:36:28.182717Z'

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

dic = {
    'images': {
        'metrics': {
            'pos': [424, 436],
            'kwargs': {'fontsize': 5, 'color': '#ffffff'},
        },
        'log_lik': {
            'pos': [424, 436],
            'kwargs': {'fontsize': 5, 'color': '#ffffff'},
        },
        'insets': [
                {
                 'rect': [190, 345, 133, 70],
                 'axes_rect': [0.8, 0.62, 0.2, 0.38],
                 'frame_path': [[0., 1.], [0., 0.], [1., 0.]],
                 'clip_path_closing': [[1., 1.]],
                },
                {
                 'rect': [220, 200, 55, 65],
                 'axes_rect': [0., 0., 0.39, 0.33],
                 'frame_path': [[1., 0.], [1., 0.45], [0.3, 1.], [0., 1.]],
                 'clip_path_closing': [[0., 0.]],
                },
        ],
    },
    'hist': {
            'num_bins': 25,
            'num_k_retained': 5, 
            'opacity': [0.3, 0.3, 0.3], 
            'zorder': [10, 5, 0],
            'color': ['#e63946', '#ee9b00', '#606c38'], 
            'linewidth': 0.75, 
            },
    'qq': {
            'zorder': [10, 5, 0],
            'color': ['blue', 'red'],
    }
}

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out) 

def create_hist_plot(ax, data, label_list, title, remove_ticks=False, color_list=None, legend_kwargs=None):
    if color_list is None:
        color_list = dic['hist']['color']
    kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)
    for (el, alpha, zorder, color, label) in zip(data, dic['hist']['opacity'], 
        dic['hist']['zorder'], color_list, label_list):
            ax.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                 facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
    ax.set_title(title)
    ax.set_xlim([0, 0.6])
    ax.set_ylim([2e-4, 150])
    ax.grid(alpha=0.3)
    ax.legend(**(legend_kwargs or {}))
    ax.set_yscale('log')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('density', labelpad=2)
    if remove_ticks: 
        ax.set_xticklabels([' '] * len(ax.get_xticklabels()))

def create_qq_plot(ax, data, label_list, title='', color_list=None, legend_kwargs=None):
    qq_xintv = [np.min(data[0][0]), np.max(data[0][0])]
    ax.plot(qq_xintv, qq_xintv, color='k', linestyle='--')
    if color_list is None:
        color_list = dic['qq']['color']
    for (osm, osr), label, color, zorder in zip(data, label_list, color_list, dic['qq']['zorder']):
        ax.plot(osm, osr, label=label, alpha=0.75, zorder=zorder, linewidth=1.75, color=color)
    abs_ylim = max(map(abs, ax.get_ylim()))
    # ax.set_xscale('symlog')
    # ax.set_yscale('symlog')
    ax.set_ylim(-abs_ylim, abs_ylim)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(**(legend_kwargs or {}))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def create_image_plot(fig, ax, image, title='', vmin=None, vmax=None, cmap='gray', interpolation='none', insets=False, insets_mark_in_orig=False, colorbar=False):
    im = ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    ax.set_title(title)
    if insets:
        for inset_spec in dic['images']['insets']:
            add_inset(fig, ax, image, **inset_spec, vmin=vmin, vmax=vmax, cmap=cmap, mark_in_orig=insets_mark_in_orig)
    if colorbar:
        cb = add_colorbar(fig, ax, im)
        if colorbar == 'invisible':
            cb.ax.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    thickness = 3.
    rect_parts = [
        ([START_0 - thickness, START_1 - thickness], END_0+1-START_0 + 2*thickness, thickness),
        ([START_0 - thickness, END_1+1], END_0+1-START_0 + 2*thickness, thickness),
        ([START_0 - thickness, START_1 - thickness], thickness, END_1+1-START_1 + 2*thickness),
        ([END_0+1, START_1 - thickness], thickness, END_1+1-START_1 + 2*thickness)]
    for rect_part in rect_parts:
        rect = matplotlib.patches.Rectangle(*rect_part, fill=True, color='#ffffff', edgecolor=None)
        ax.add_patch(rect)
    return im

def add_colorbar(fig, ax, im):
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="4%", pad="2%")
    cb = fig.colorbar(im, cax=cax)
    cax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    return cb

def add_inset(fig, ax, image, axes_rect, rect, cmap='gray', vmin=None, vmax=None, interpolation='none', frame_color='#aa0000', frame_path=None, clip_path_closing=None, mark_in_orig=False, origin='upper'):
    ip = InsetPosition(ax, axes_rect)
    axins = matplotlib.axes.Axes(fig, [0., 0., 1., 1.])
    axins.set_axes_locator(ip)
    fig.add_axes(axins)
    slice0 = slice(rect[0], rect[0]+rect[2])
    slice1 = slice(rect[1], rect[1]+rect[3])
    inset_image = image[slice0, slice1]
    inset_image_handle = axins.imshow(
            inset_image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.patch.set_visible(False)
    for spine in axins.spines.values():
        spine.set_visible(False)
    if frame_path is None:
        frame_path = [[0., 0.], [1., 0.], [0., 1.], [1., 1]]
    if frame_path:
        frame_path_closed = frame_path + (clip_path_closing if clip_path_closing is not None else [])
        if mark_in_orig:
            scalex, scaley = rect[3], rect[2]
            offsetx, offsety = rect[1], (image.shape[0]-(rect[0]+rect[2]) if origin == 'upper' else rect[0])
            y_trans = matplotlib.transforms.Affine2D().scale(1., -1.).translate(0., image.shape[0]-1) if origin == 'upper' else matplotlib.transforms.IdentityTransform()
            trans_data = matplotlib.transforms.Affine2D().scale(scalex, scaley).translate(offsetx, offsety) + y_trans + ax.transData
            x, y = [*zip(*(frame_path_closed + [frame_path_closed[0]]))]
            ax.plot(x, y, transform=trans_data, color=frame_color, linestyle='dashed', linewidth=1.)
        axins.plot(
                *np.array(frame_path).T,
                transform=axins.transAxes,
                color=frame_color,
                solid_capstyle='butt')
        inset_image_handle.set_clip_path(matplotlib.path.Path(frame_path_closed),
                transform=axins.transAxes)
        inset_image_handle.set_clip_on(True)

def add_metrics(ax, recon, image, pos=None, as_xlabel=True, **kwargs):
    psnr = PSNR(recon, image)
    ssim = SSIM(recon, image)
    s_psnr = 'PSNR: ${:.3f}$\\,dB'.format(psnr)
    s_ssim = 'SSIM: ${:.4f}$'.format(ssim)
    if as_xlabel:
        ax.set_xlabel(s_psnr + ';\;' + s_ssim)
    else:
        if pos is None:
            pos = dic['images']['metrics']['pos']
        kwargs.update(dic['images']['metrics'].get('kwargs', {}))
        ax.text(*pos, s_psnr + '\n' + s_ssim, ha='right', va='top', **kwargs)

def add_log_lik(ax, log_lik, pos=None, as_xlabel=True, **kwargs):
    s = 'log-likelihood: ${:.4f}$'.format(log_lik)
    if as_xlabel:
        ax.set_xlabel(s)
    else:
        if pos is None:
            pos = dic['images']['log_lik']['pos']
        kwargs.update(dic['images']['log_lik'].get('kwargs', {}))
        ax.text(*pos, s, ha='right', va='top', **kwargs)


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

def normalized_error_for_qq_plot(recon, image, std):
    normalized_error = (recon - image) / std
    return normalized_error

@hydra.main(config_path='../../cfgs', config_name='config')  # to enable get_original_cwd
def plot_walnut(cfg):
# def plot_walnut():

    fs_m1 = 6  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 12  #  figure titles

    color_abs_error = '#e63946'
    color_mll = '#5555ff'
    color_map = '#5a6c17'
    color_mcdo = '#ee9b00'

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs_p1)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)        # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title
    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('text.latex', preamble='\\usepackage{amsmath}')

    full_run_path_mll = os.path.join(DIR_PATH, run_path_mll)
    full_run_path_map = os.path.join(DIR_PATH, run_path_map)
    full_run_path_mcdo = os.path.join(DIR_PATH, run_path_mcdo)

    subtract_noise_correction_term = True

    cfg = OmegaConf.load(os.path.join(full_run_path_mll, '.hydra', 'config.yaml'))  # use mll config, the settings relevant in this script should be same for all methods
    assert cfg.density.block_size_for_approx == BLOCK_SIZE

    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-20T00:56:37.631314Z'  # TODO remove
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-21T13:32:36.795812Z'  # no sigma_y override TODO remove
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-24T20:35:18.987288Z'  # TODO remove
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-26T16:28:47.864713Z'  # 1x1
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-26T16:33:15.970045Z'  # 2x2
    if cfg.get('load_from_previous_run_path'):
        plot_data = np.load(os.path.join(cfg.load_from_previous_run_path, 'plot_data.npz'))
        image = plot_data['image']; observation_2d = plot_data['observation_2d']; filtbackproj = plot_data['filtbackproj']; recon = plot_data['recon']; recon_mcdo = plot_data['recon_mcdo']
        abs_error = plot_data['abs_error']; abs_error_mcdo = plot_data['abs_error_mcdo']
        log_lik_mll = plot_data['log_lik_mll']; log_lik_map = plot_data['log_lik_map']; log_lik_mcdo = plot_data['log_lik_mcdo']
        std_pred_mll = plot_data['std_pred_mll']; std_pred_map = plot_data['std_pred_map']; std_pred_mcdo = plot_data['std_pred_mcdo']
        inner_inds = plot_data['inner_inds']
        noise_correction_term_mll = plot_data['noise_correction_term_mll']; noise_correction_term_map = plot_data['noise_correction_term_map']; noise_correction_term_mcdo = plot_data['noise_correction_term_mcdo']
    else:
        ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
        image, observation_2d, filtbackproj, recon, abs_error = collect_reconstruction_data(full_run_path_mll, cfg, ray_trafos)
        noise_correction_term_mll = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=collect_log_noise_model_variance_obs(full_run_path_mll))
        noise_correction_term_map = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=collect_log_noise_model_variance_obs(full_run_path_map))
        noise_correction_term_mcdo = get_noise_correction_term(ray_trafos, log_noise_model_variance_obs=0.)
        inner_block_idx_list = get_inner_block_indices(BLOCK_SIZE)
        all_block_mask_inds = get_image_block_mask_inds(IM_SHAPE, cfg.density.block_size_for_approx)
        inner_block_mask_inds = np.array(all_block_mask_inds)[inner_block_idx_list]
        inner_inds = np.concatenate(inner_block_mask_inds)

        log_lik_mll, std_pred_mll = collect_dip_bayes_data(full_run_path_mll, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_mll)
        log_lik_map, std_pred_map = collect_dip_bayes_data(full_run_path_map, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_map)
        log_lik_mcdo, recon_mcdo, abs_error_mcdo, std_pred_mcdo = collect_mcdo_data(full_run_path_mcdo, cfg, inner_block_idx_list, all_block_mask_inds, noise_correction_term=noise_correction_term_mcdo)

        np.savez('plot_data.npz',
            **{'image': image, 'observation_2d': observation_2d, 'filtbackproj': filtbackproj, 'recon': recon, 'recon_mcdo': recon_mcdo,
            'abs_error': abs_error, 'abs_error_mcdo': abs_error_mcdo,
            'log_lik_mll': log_lik_mll, 'log_lik_map': log_lik_map, 'log_lik_mcdo': log_lik_mcdo,
            'std_pred_mll': std_pred_mll, 'std_pred_map': std_pred_map, 'std_pred_mcdo': std_pred_mcdo,
            'inner_inds': inner_inds,
            'noise_correction_term_mll': noise_correction_term_mll, 'noise_correction_term_map': noise_correction_term_map, 'noise_correction_term_mcdo': noise_correction_term_mcdo})

    # TODO also subtract eps

    inner_image = image.reshape(-1)[inner_inds]
    inner_recon = recon.reshape(-1)[inner_inds]
    inner_recon_mcdo = recon_mcdo.reshape(-1)[inner_inds]
    inner_abs_error = abs_error.reshape(-1)[inner_inds]
    inner_abs_error_mcdo = abs_error_mcdo.reshape(-1)[inner_inds]
    inner_std_pred_mll = std_pred_mll.reshape(-1)[inner_inds]
    inner_std_pred_map = std_pred_map.reshape(-1)[inner_inds]
    inner_std_pred_mcdo = std_pred_mcdo.reshape(-1)[inner_inds]

    # qq_err_mll = normalized_error_for_qq_plot(inner_recon, inner_image, inner_std_pred_mll)
    # qq_err_map = normalized_error_for_qq_plot(inner_recon, inner_image, inner_std_pred_map)
    # qq_err_mcdo = normalized_error_for_qq_plot(inner_recon_mcdo, inner_image, inner_std_pred_mcdo)

    images_dir = os.path.join(IMAGES_DIR)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)


    fig, axs = plt.subplots(1, 6, figsize=(14, 2.15), gridspec_kw={
        'width_ratios': [1., 1., 1., 1., 0.2, 1.25],  # includes spacer columns
        'wspace': 0.1, 'hspace': 0.25})

    # nan parts black
    std_pred_mll[np.isnan(std_pred_mll)] = 0.
    std_pred_map[np.isnan(std_pred_map)] = 0.
    std_pred_mcdo[np.isnan(std_pred_mcdo)] = 0.

    vmin_std, vmax_std = min([np.min(abs_error), np.min(std_pred_map)]), np.max(abs_error)  # max([np.max(abs_error), np.max(std_pred_map)])

    create_image_plot(fig, axs[0], image, title='${\mathbf{x}}$', vmin=0., insets=True, insets_mark_in_orig=True)
    axs[1].set_ylabel('Bayes DIP', fontsize=fs_p1)
    create_image_plot(fig, axs[1], abs_error, title='$|{\mathbf{x} - \mathbf{x}^*}|$', vmin=0., vmax=vmax_std, insets=True)  #, vmin=vmin_std, vmax=vmax_std)
    create_image_plot(fig, axs[2], std_pred_mll, vmin=0., vmax=vmax_std, title='std-dev', insets=True)  # , vmin=vmin_std, vmax=vmax_std)
    axs[2].set_ylabel('MLL', fontsize=fs_p1)
    add_log_lik(axs[2], log_lik_mll)
    create_image_plot(fig, axs[3], std_pred_map, vmin=0., vmax=vmax_std, title='std-dev', insets=True, colorbar=True)  # , vmin=vmin_std, vmax=vmax_std)
    axs[3].set_ylabel('TV-MAP', fontsize=fs_p1)
    add_log_lik(axs[3], log_lik_map)

    # spacer
    axs[4].remove()

    create_hist_plot(
        axs[5],
        (inner_abs_error, inner_std_pred_mll, inner_std_pred_map), 
        ['$|{\mathbf{x} - \mathbf{x}^*}|$', 'std-dev -- Bayes DIP (MLL)', 'std-dev -- Bayes DIP (TV-MAP)'],
        'marginal std-dev',
        False,
        color_list=[color_abs_error, color_mll, color_map],
        legend_kwargs={'loc': 'upper right', 'bbox_to_anchor': (1.015, 0.99)},
        )
    # axs[5].set_aspect(0.125)

    # qq_host_axis = axs[1, 7]
    # qq_axes_rect = [0.48, 0.44, 0.52, 0.65]
    # ip = InsetPosition(qq_host_axis, qq_axes_rect)
    # ax_qq = matplotlib.axes.Axes(fig, [0., 0., 1., 1.])
    # ax_qq.set_axes_locator(ip)
    # ax_qq.set_clip_on(False)
    # ax_qq.set_clip_on(False)
    # border_0, border_1 = 0.185, 0.25
    # corner_crop_0, corner_crop_1 = 0.25, 0.3
    # qq_background = matplotlib.patches.Polygon(
    #         [[qq_axes_rect[0] - border_0, qq_axes_rect[1] - border_1 + corner_crop_1],
    #          [qq_axes_rect[0] - border_0 + corner_crop_0, qq_axes_rect[1] - border_1],
    #          [qq_axes_rect[0] + qq_axes_rect[2], qq_axes_rect[1] - border_1],
    #          [qq_axes_rect[0] + qq_axes_rect[2], qq_axes_rect[1] + qq_axes_rect[3]],
    #          [qq_axes_rect[0] - border_0, qq_axes_rect[1] + qq_axes_rect[3]]],
    #         fill=True, color='#ffffff', edgecolor=None, transform=qq_host_axis.transAxes, zorder=3)
    # qq_background.set_clip_on(False)
    # qq_host_axis.add_patch(qq_background)
    # fig.add_axes(ax_qq)

    # # osm_mll, osr_mll = scipy.stats.probplot(qq_err_mll, fit=False)
    # osm_map, osr_map = scipy.stats.probplot(qq_err_map, fit=False)
    # osm_mcdo, osr_mcdo = scipy.stats.probplot(qq_err_mcdo, fit=False)
    # create_qq_plot(ax_qq,
    #     [(osm_map, osr_map), (osm_mcdo, osr_mcdo)],
    #     ['Bayes DIP', 'DIP-MCDO'],
    #     color_list=[color_map, color_mcdo],
    #     legend_kwargs={'loc': 'lower right', 'bbox_to_anchor': (1., 0.)})
    # # ax_qq.set_aspect(np.diff(ax_qq.get_xlim())/np.diff(ax_qq.get_ylim()))
    # ax_qq.add_patch(matplotlib.patches.Rectangle([0.05, 0.95], 0.9, 0.05,
    #         fill=True, color='#ffffff', edgecolor=None, transform=ax_qq.transAxes, zorder=3))
    # ax_qq.set_title('calibration: Q-Q', y=0.93)
    # ax_qq.set_xlabel('prediction quantiles', labelpad=2)
    # ax_qq.set_ylabel('error quantiles', labelpad=2)

    fig.savefig(os.path.join(IMAGES_DIR, f'walnut_mll_vs_map_{BLOCK_SIZE}x{BLOCK_SIZE}.pdf'), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(IMAGES_DIR, f'walnut_mll_vs_map_{BLOCK_SIZE}x{BLOCK_SIZE}.png'), bbox_inches='tight', pad_inches=0., dpi=600)
    plt.show()

if __name__ == "__main__": 
    plot_walnut()
