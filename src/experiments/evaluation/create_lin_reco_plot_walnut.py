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

DIR_PATH='/localdata/experiments/dip_bayesian_ext/'
IMAGES_DIR='./images_walnut'

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
}

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

def collect_dip_bayes_lin_reco(path):
    load_path = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml')).density.compute_single_predictive_cov_block.load_path
    lin_pred = torch.load(os.path.join(DIR_PATH, load_path, 'linearized_weights_0.pt'))['linearized_prediction']
    return lin_pred.detach().cpu().numpy().squeeze()

@hydra.main(config_path='../../cfgs', config_name='config')  # to enable get_original_cwd
def table_walnut(cfg):
# def table_walnut():

    fs_m1 = 6  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 12  #  figure titles

    color_abs_error = '#e63946'
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

    cfg = OmegaConf.load(os.path.join(full_run_path_mll, '.hydra', 'config.yaml'))  # use mll config, the settings relevant in this script should be same for all methods

    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-24T20:43:34.659765Z/'  # TODO remove
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-26T16:45:35.641681Z'  # 1x1
    # cfg.load_from_previous_run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-26T16:46:30.466402Z'  # 2x2
    if cfg.get('load_from_previous_run_path'):
        plot_data = np.load(os.path.join(cfg.load_from_previous_run_path, 'plot_data.npz'))
        image = plot_data['image']; recon = plot_data['recon']; lin_pred_mll = plot_data['lin_pred_mll']; lin_pred_map = plot_data['lin_pred_map']
        inner_inds = plot_data['inner_inds']
    else:
        ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
        image, observation_2d, filtbackproj, recon, abs_error = collect_reconstruction_data(full_run_path_mll, cfg, ray_trafos)
        lin_pred_mll = collect_dip_bayes_lin_reco(full_run_path_mll)
        lin_pred_map = collect_dip_bayes_lin_reco(full_run_path_map)
        inner_block_idx_list = get_inner_block_indices(BLOCK_SIZE)
        all_block_mask_inds = get_image_block_mask_inds(IM_SHAPE, cfg.density.block_size_for_approx)
        inner_block_mask_inds = np.array(all_block_mask_inds)[inner_block_idx_list]
        inner_inds = np.concatenate(inner_block_mask_inds)
        np.savez('plot_data.npz',
            **{'image': image, 'recon': recon, 'lin_pred_mll': lin_pred_mll, 'lin_pred_map': lin_pred_map,
               'inner_inds': inner_inds})

    inner_image = image.reshape(-1)[inner_inds]
    inner_recon = recon.reshape(-1)[inner_inds]
    inner_lin_pred_mll = lin_pred_mll.reshape(-1)[inner_inds]
    inner_lin_pred_map = lin_pred_map.reshape(-1)[inner_inds]  # should be the same as for mll up to run variability


    images_dir = os.path.join(IMAGES_DIR)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    fig, axs = plt.subplots(1, 6, figsize=(14, 2.5), gridspec_kw={
        'width_ratios': [1., 1., 1., 0.1, 1., 1.],  # includes spacer column
        'wspace': 0.01, 'hspace': 0.25})
    axs[3].remove()
    axs = axs[[0, 1, 2, 4, 5]]

    abs_error = np.abs(recon - image)
    abs_error_lin_pred_mll = np.abs(lin_pred_mll - image)

    vmax_images = max(np.max(image), np.max(recon), np.max(lin_pred_mll))
    vmax_errors = max(np.max(abs_error), np.max(abs_error_lin_pred_mll))

    create_image_plot(fig, axs[0], image, title='${\mathbf{x}}$', vmin=0., vmax=vmax_images, insets=True, insets_mark_in_orig=True)
    create_image_plot(fig, axs[1], recon, title='DIP: ${\mathbf{f}^\star}$', vmin=0., vmax=vmax_images, insets=True)
    add_metrics(axs[1], recon[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1])
    create_image_plot(fig, axs[2], lin_pred_mll, title='linearized: ${\mathbf{f}^\star}$', vmin=0., vmax=vmax_images, insets=True, colorbar=True)
    add_metrics(axs[2], lin_pred_mll[START_0:END_0, START_1:END_1], image[START_0:END_0, START_1:END_1])
    create_image_plot(fig, axs[3], abs_error, title='DIP: $|{\mathbf{x} - \mathbf{f}^\star}|$', vmin=0., vmax=vmax_errors, insets=True)
    create_image_plot(fig, axs[4], abs_error_lin_pred_mll, title='linearized: $|{\mathbf{x} - \mathbf{f}^\star}|$', vmin=0., vmax=vmax_errors, insets=True, colorbar=True)

    fig.savefig(os.path.join(IMAGES_DIR, f'walnut_lin_reco_{BLOCK_SIZE}x{BLOCK_SIZE}.pdf'), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(IMAGES_DIR, f'walnut_lin_reco_{BLOCK_SIZE}x{BLOCK_SIZE}.png'), bbox_inches='tight', pad_inches=0., dpi=600)
    plt.show()

if __name__ == "__main__": 
    table_walnut()
