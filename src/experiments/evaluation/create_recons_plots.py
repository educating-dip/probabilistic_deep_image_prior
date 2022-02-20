import sys
sys.path.append('../')

import os
import numpy as np 
import numpy as np
import torch
import tensorly as tl
tl.set_backend('pytorch')
import matplotlib
import scipy.stats
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from deep_image_prior.utils import PSNR, SSIM
from dataset.utils import get_standard_ray_trafos
from dataset import extract_trafos_as_matrices

DIRPATH='src/experiments/evaluation/'  # TODO insert absolute path if needed

set_xlim_dict = {
    (5, 0.05): 0.90, 
    (5, 0.1): 0.95, 
    (10, 0.05): 0.45, 
    (10, 0.1): 0.6,
    (20, 0.05): 0.25,
    (20, 0.1): 0.35, 
    (30, 0.05): 0.25,
    (30, 0.1): 0.25
}

dic = {'images':
        {   
            'num': 50, 
            'n_rows': 3,
            'n_cols': 6,
            'figsize': (14, 7),
            'idx_to_norm': [2, 3, 4],
            'idx_vmax_vmin_1_0': [0, 1, 6, 7, 13],
            'idx_add_cbar': [4, 8, 9, 14, 15],
            'idx_add_psnr': [1, 7, 13],
            'idx_add_qq_plot': [10, 16],
            'add_lateral_title': {1: 'Bayes DIP', 7: 'DIP-MCDO', 13: 'DIP-SGLD', 3: 'MLL', 4:'TV-MAP', 'idx': [1, 3, 4, 7, 13]}, 
            'idx_add_test_loglik': [3, 4, 9, 15],
            'idx_test_log_lik': {3:0, 4:1, 9:2, 15:3},
            'idx_hist_insert': [5, 11, 17],
            'idx_remove_ticks': [5, 11,],
        },
        'hist': 
        {
            'num_bins': 25,
            'num_k_retained': 5, 
            'opacity': [0.3, 0.3, 0.3], 
            'zorder': [10, 5, 0],
            'color': {5: ['#e63946', '#35DCDC', '#5A6C17'], 11:['#e63946', '#EE9B00'], 17:['#e63946', '#781C68']}, 
            'linewidth': 0.75, 
            }, 
        'qq': 
        {
            'zorder': [10, 5, 0],
            'color': {10: ['#35DCDC', '#5A6C17', '#EE9B00'], 16:  ['#35DCDC', '#5A6C17', '#781C68']},
            }
}

def create_qq_plot(ax, data, label_list, title='', color_list=None, legend_kwargs=None):
    qq_xintv = [np.min(data[0][0]), np.max(data[0][0])]
    ax.plot(qq_xintv, qq_xintv, color='k', linestyle='--')
    if color_list is None:
        color_list = dic['qq']['color'][-1]
    for (osm, osr), label, color, zorder in zip(data, label_list, color_list, dic['hist']['zorder']):
        ax.plot(osm, osr, label=label, alpha=0.75, zorder=zorder, linewidth=1.75, color=color)
    abs_ylim = max(map(abs, ax.get_ylim()))
    ax.set_ylim(-abs_ylim, abs_ylim)
    ax.set_title(title)
    ax.set_xlabel('prediction quantiles')
    ax.set_ylabel('error quantiles')
    ax.grid(alpha=0.3)
    ax.legend(**(legend_kwargs or {}), loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def kmnist_image_fig_subplots(data, loglik, filename, titles):

    fs_m1 = 8  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 24 #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=r'\usepackage{amsmath}'

    fig, axs = plt.subplots(dic['images']['n_rows'], dic['images']['n_cols'], figsize=dic['images']['figsize'],
          facecolor='w', edgecolor='k', constrained_layout=True)
    
    for i, (el, ax, title) in enumerate(zip(data, axs.flatten(), titles)):
        if i in dic['images']['idx_hist_insert']:
            kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)
            for (el, alpha, zorder, color, label) in zip(el[0], dic['hist']['opacity'], 
                dic['hist']['zorder'], dic['hist']['color'][i], el[1]):
                    ax.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                        facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
            ax.set_title(title, y=1.01)
            ax.set_xlim([0, set_xlim_dict[(num_angles, stddev)]])
            ax.set_ylim([0.09, 90])
            ax.set_yscale('log')
            ax.set_ylabel('density')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend()
            ax.grid(alpha=0.3)
            if i in dic['images']['idx_remove_ticks']: 
                ax.set_xticklabels([' ', ' ', ' ', ' '])
        elif i in dic['images']['idx_add_qq_plot']:
            osm_mll, osr_mll = scipy.stats.probplot(el[0].flatten(), fit=False)
            osm_map, osr_map = scipy.stats.probplot(el[1].flatten(), fit=False)
            osm_baseline, osr_baseline = scipy.stats.probplot(el[2].flatten(), fit=False)
            create_qq_plot(ax,
                [(osm_mll, osr_mll), (osm_map, osr_map), (osm_baseline, osr_baseline)],
                ['Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'DIP-MCDO'] if i == 10 else ['Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'DIP-SGLD'],
                title=title, color_list=dic['qq']['color'][i])
            ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == dic['images']['idx_add_qq_plot'][0]: 
                ax.set_xticklabels([' ', ' ', ' ', ' '])
        else:
            if i in dic['images']['idx_vmax_vmin_1_0']: 
                im = ax.imshow(el, cmap='gray', vmin=0, vmax=1) 
            if i in dic['images']['idx_to_norm']:
                vmax = np.max(data[dic['images']['idx_to_norm'][0]:dic['images']['idx_to_norm'][-1]])
                vmin = np.min(data[dic['images']['idx_to_norm'][0]:dic['images']['idx_to_norm'][-1]])
                im = ax.imshow(el, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(el, cmap='gray', )
            ax.set_title(title)
            if i in dic['images']['add_lateral_title']['idx']:
                ax.set_ylabel(dic['images']['add_lateral_title'][i])
        
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            if i in dic['images']['idx_add_cbar']: 
                fig.colorbar(im, ax=ax, shrink=0.90)
                
            if i in dic['images']['idx_add_psnr']:
                psnr = PSNR(el.flatten(), data[0].flatten())
                ssim = SSIM(el.flatten(), data[0].flatten())
                s_psnr = 'PSNR: ${:.3f}$\\,dB'.format(psnr)
                s_ssim = 'SSIM: ${:.4f}$'.format(ssim)
                ax.set_xlabel(s_psnr + ';\;' + s_ssim)
    
            if i in dic['images']['idx_add_test_loglik']:
                ax.set_xlabel('log-likelihood: ${:.4f}$'.format(loglik[dic['images']['idx_test_log_lik'][i]]))

    fig.savefig(filename + '.png', dpi=100)
    fig.savefig(filename + '.pdf')

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out) 

def gather_data_from_bayes_dip(idx):
    
    runs = OmegaConf.load(os.path.join(DIRPATH, 'kmnist_refined_tv_strength.yaml')) # 'runs.yaml'))
    path_to_data = runs[num_angles][stddev] # runs.kmnist[num_angles][stddev][0]['path'] # selecting first run in yaml file [0]
    recon_data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)
    log_lik_data = np.load(os.path.join(path_to_data, 'test_log_lik_info_{}.npz'.format(idx)),  allow_pickle=True)['test_log_lik'].item()

    example_image = recon_data['image'].squeeze()
    filtbackproj = recon_data['filtbackproj'].squeeze()
    observation = recon_data['observation'].squeeze()
    recon = recon_data['recon'].squeeze()
    abs_error = np.abs(example_image - recon)
    pred_cov_matrix_mll = recon_data['model_post_cov_no_predcp']
    pred_cov_matrix_tv_map = recon_data['model_post_cov_predcp']
    std_pred_mll = np.sqrt(np.diag(pred_cov_matrix_mll)).reshape(28, 28)
    std_pred_tv_map = np.sqrt(np.diag(pred_cov_matrix_tv_map)).reshape(28, 28)
    
    return (example_image, filtbackproj, observation, recon, abs_error, std_pred_mll, std_pred_tv_map,
                pred_cov_matrix_mll, pred_cov_matrix_tv_map, log_lik_data['test_loglik_MLL'], log_lik_data['test_loglik_type-II-MAP'])


def estimate_noise_in_x(cfg):
    
    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=False, return_op_mat=True)
    trafos = extract_trafos_as_matrices(ray_trafos)
    trafo = trafos[0]
    trafo_T_trafo = trafo.T @ trafo
    U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
    lik_hess_inv_diag_mean = (Vh.T @ torch.diag(1/S) @ U.T).diag().mean() 
    return lik_hess_inv_diag_mean.numpy()


def gather_data_from_baseline_dip(idx, run_path):
    
    runs = OmegaConf.load(os.path.join(DIRPATH, run_path))
    path_to_data = runs[num_angles][stddev]
    conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
    noise_in_x = estimate_noise_in_x(conf)
    data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)

    example_image = data['image'].squeeze()
    recon = data['recon'].squeeze().reshape(28, 28)
    abs_error = np.abs(example_image - recon)
    std_pred = np.clip(data['std'].reshape(28, 28) ** 2 - noise_in_x, a_min=0, a_max=np.inf) **.5
    return (recon, abs_error, std_pred, data['test_log_likelihood'].item())

def normalized_error_for_qq_plot(recon, image, std):
    normalized_error = (recon - image) / std
    return normalized_error

if __name__ == "__main__":


    for angles in [5, 10, 20, 30]:
        for noise in [0.05, 0.1]: 
            global num_angles
            num_angles = angles
            global stddev
            stddev = noise

            for idx in range(dic['images']['num']):

                (example_image, filtbackproj, observation, recon, abs_error, std_pred_mll, std_pred_tv_map,
                        pred_cov_matrix_mll, pred_cov_matrix_map, test_log_lik_mll, test_log_lik_tv_map) = gather_data_from_bayes_dip(idx)
                folder_name = 'kmnist_num_angles_{}_stddev_{}'.format(num_angles, stddev)
                dir_path = os.path.join('./', 'images', folder_name)

                (mcdo_recon, mcdo_abs_error, mcdo_std_pred, mcdo_test_log_lik) = gather_data_from_baseline_dip(idx, 'kmnist_mcdo_baseline_bw_005.yaml')
                (sgld_recon, sgld_abs_error, sgld_std_pred, sgld_test_log_lik) = gather_data_from_baseline_dip(idx, 'kmnist_sgld_baseline_bw_005.yaml')

                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)

                qq_err_mll = normalized_error_for_qq_plot(recon, example_image, std_pred_mll)
                qq_err_map = normalized_error_for_qq_plot(recon, example_image, std_pred_tv_map)
                qq_err_mcdo = normalized_error_for_qq_plot(mcdo_recon, example_image, mcdo_std_pred)
                qq_err_sgld = normalized_error_for_qq_plot(sgld_recon, example_image, sgld_std_pred)

                kmnist_image_fig_subplots( 
                    (
                    example_image, recon, abs_error, std_pred_mll, std_pred_tv_map, 
                    (
                        (abs_error, std_pred_mll, std_pred_tv_map),
                        ['$|{\mathbf{x} - \mathbf{x}^*}|$', 'std-dev (MLL)', 'std-dev (TV-MAP)']
                    ),
                    filtbackproj,
                    mcdo_recon, mcdo_abs_error, mcdo_std_pred,
                    ( qq_err_mll, qq_err_map, qq_err_mcdo) , 
                    (
                            (mcdo_abs_error, mcdo_std_pred),  
                            ['$|{\mathbf{x} - \mathbf{x}^*}|$', 'DIP-MCDO']
                    ),
                    np.transpose(observation),
                    sgld_recon, sgld_abs_error, sgld_std_pred,
                    ( qq_err_mll, qq_err_map, qq_err_sgld), 
                    (
                            (sgld_abs_error, sgld_std_pred),  
                            ['$|{\mathbf{x} - \mathbf{x}^*}|$', 'DIP-SGLD']
                    )
                    ),
                    (test_log_lik_mll, test_log_lik_tv_map, mcdo_test_log_lik, sgld_test_log_lik),
                    dir_path + '/main_{}'.format(idx), 
                    ['${\mathbf{x}}$','${\mathbf{x}^*}$', '${|\mathbf{x} - \mathbf{x}^*|}$', 
                    'std-dev', 'std-dev',  'marginal std-dev','FBP (i.e.\ ${\\textnormal{A}}^\dagger \mathbf{y}_{\delta})$', '',
                    '', '', 'Calibration: Q-Q', '',
                    '$\mathbf{y}_{\delta}$', '', '', '', '', '']
                    )
