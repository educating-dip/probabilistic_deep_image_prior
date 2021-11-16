import os
import numpy as np 
import yaml
import bios
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt 
from matplotlib import ticker
from copy import deepcopy

DIR_PATH='/media/chen/Res/dip_bayesian_ext/src/experiments' 

run_paths = bios.read(os.path.join(DIR_PATH, 'evaluation/runs.yaml'))
name = 'kmnist_sparse_20' # kmnist_sparse_20
idx = 1 # [0, 1]
exp_name = 'calibration_uncertainty_estimates'

dic = {'images':
        {   'num': 10, 
            'n_rows': 2,
            'n_cols': 3,
            'figsize': (6, 4),
            'idx_to_norm': 3,
            'idx_add_cbar': [2, 5],
            'idx_add_psnr': [1, 2],
            'idx_add_min_max': [3, 4, 5], 
            'idx_add_test_loglik': [4, 5], 
            'text_insert':
            {
                'psnr': [4.75, 29], 
                'min_max': [-2, 16.85], 
                'test_loglik': [10, 29], 
                'fontsize': 7.5,
            }
        },
    'hist': {
            'num_bins': 25,
            'num_k_retained': 5, 
            'opacity': [0.3, 0.3, 0.3], 
            'zorder': [10, 5, 0],
            'color': ['#e63946', '#ee9b00', '#606c38'], 
            'linewidth': 2.25, 
            },
    'eigenUQ': 
    {
        'n_rows': 2,
        'n_cols': 5,
            
    }
}

def create_eigs_uncertainty_subplots(abs_error, v_mll, v_map, filename):

    fs_m1 = 6  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 15  #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)  # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    fig, axs = plt.subplots(dic['eigenUQ']['n_rows'], dic['eigenUQ']['n_cols'],
        figsize=(8, 3), gridspec_kw = {'wspace':0, 'hspace':0}, facecolor='w', edgecolor='k', constrained_layout=True)
    axs = axs.flatten()
    for i in range(v_mll.shape[1]):
        if i == 0:
            axs[i].text(-6.5, 14, 
                'MLL', rotation=90, horizontalalignment='center', 
                verticalalignment='center', color='black')

        axs[i].set_title('v$_{}$'.format(i), y=1.15)
        im = axs[i].imshow(v_mll[:, i].reshape(28,28),
            vmin=-np.max(np.abs(v_mll)), vmax=np.max(np.abs(v_mll)), cmap='RdGy')
        axs[i].set_axis_off()

    shift = v_mll.shape[1]
    for i in range(v_map.shape[1]):
        if i == 0:
            axs[i + shift].text(-6.5, 14, 
                'Type-II MAP', rotation=90, horizontalalignment='center', 
                verticalalignment='center', color='black')
        im = axs[i + shift].imshow(v_map[:, i].reshape(28,28), 
            vmin=-np.max(np.abs(v_map)), vmax=np.max(np.abs(v_map)), cmap='RdGy')
        axs[i + shift].set_axis_off()

    cb = fig.colorbar(im, ax=[axs[2], axs[-1]], shrink=0.95, aspect=40)
    cb.ax.locator_params(nbins=10)

    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')

def create_images_fig_subplots(data, loglik, filename, titles):

    fs_m1 = 6  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 15  #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title


    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    fig, axs = plt.subplots(dic['images']['n_rows'], dic['images']['n_cols'], figsize=dic['images']['figsize'],
          facecolor='w', edgecolor='k', constrained_layout=True)

    vmax = np.max(data[dic['images']['idx_to_norm']:])
    vmin = np.min(data[dic['images']['idx_to_norm']:])
    k = 0 
    for i, (el, ax, title) in enumerate(zip(data, axs.flatten(), titles)):
        im = ax.imshow(el, cmap='gray', vmin=0, vmax=1) if i < dic['images']['idx_to_norm'] \
            else ax.imshow(el, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_axis_off()
        ax.set_title(title)
        if i in dic['images']['idx_add_cbar']: 
            fig.colorbar(im, ax=ax, shrink=0.95)
        
        if i in dic['images']['idx_add_psnr']: 
            psnr = 10 * np.log10( 1 / np.mean( (data[0].flatten() - el.flatten()) **2))
            ax.text(dic['images']['text_insert']['psnr'][0], dic['images']['text_insert']['psnr'][1], 
                'psnr: {:.4f}'.format(psnr), horizontalalignment='center', verticalalignment='center', 
                    color='black', fontsize=dic['images']['text_insert']['fontsize'])

        if i in dic['images']['idx_add_min_max']: 
            ax.text(dic['images']['text_insert']['min_max'][0], dic['images']['text_insert']['min_max'][1], 
                'max: {:.1e}, min: {:.1e}'.format(el.max(), el.min()), rotation=90, horizontalalignment='center', 
                verticalalignment='center', color='black', fontsize=dic['images']['text_insert']['fontsize'])
        if i in dic['images']['idx_add_test_loglik']:
            ax.text(dic['images']['text_insert']['test_loglik'][0], dic['images']['text_insert']['test_loglik'][1], 
                'test log-likelihood: {:.4f}'.format(loglik[k]), horizontalalignment='center', verticalalignment='center',
                color='black', fontsize=dic['images']['text_insert']['fontsize'])
            k += 1
    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out) 

def create_hist_plot(data, filename, label_list, title):

    fs_m1 = 15  # for figure ticks
    fs = 20  # for regular figure text
    fs_p1 = 20  #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)    # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)  # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    fig, axs = plt.subplots(1, 1, figsize=(6,6),  facecolor='w', 
        edgecolor='k', constrained_layout=True)

    kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)

    for (el, alpha, zorder, color, label) in zip(data, dic['hist']['opacity'], 
        dic['hist']['zorder'], dic['hist']['color'], label_list):
            axs.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                 facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
    axs.set_title(title, y=1.01)
    axs.set_xlim([0, 0.75])
    axs.grid(alpha=0.3)
    axs.legend()
    axs.set_yscale('log')
    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


def collect_test_log_lik(path, idx):
  
    test_log_lik_info = np.load(os.path.join(DIR_PATH, path, \
        'test_log_lik_info_{}.npz'.format(idx)), allow_pickle=True)
    return (test_log_lik_info['test_log_lik'].item()['MLL'], \
        test_log_lik_info['test_log_lik'].item()['type-II-MAP'] )

def collect_reconstruction_data(path, idx):
    
    recon_info = np.load(os.path.join(DIR_PATH, path, 'recon_info_{}.npz'.format(idx)))

    image = \
        recon_info['image'].squeeze()
    filtbackproj = \
        recon_info['filtbackproj'].squeeze()
    recon = \
        recon_info['recon'].squeeze()
    abs_error = \
        np.abs(image - recon).squeeze()
    covariance_matrix_mll = \
         recon_info['model_post_cov_no_predCP'] 
    covariance_matrix_map = \
        recon_info['model_post_cov_w_PredCP']
    std_pred_mll = \
        np.sqrt(np.diag(covariance_matrix_mll)).reshape(28, 28)
    std_pred_map = \
        np.sqrt(np.diag(covariance_matrix_map)).reshape(28, 28)

    return image, filtbackproj, recon, abs_error, std_pred_mll, std_pred_map, covariance_matrix_mll, covariance_matrix_map

def extract_eigenvectors(data, k):
    
    s, v = np.linalg.eigh(data)
    lev_score = ( (v[:, -10:])**2 ).sum(axis=1) 
    lev_score = ( lev_score - np.min(lev_score) ) / ( np.max(lev_score) - np.min(lev_score) )
    return np.sqrt(s), v[:, -k:], lev_score.reshape(28, 28)

for i in range(dic['images']['num']):

    name_dir, path_to_data = run_paths[exp_name][name][idx]['name'], \
        run_paths[exp_name][name][idx]['path']
    image, filtbackproj, recon,\
    abs_error, std_pred_mll, std_pred_map, \
    covariance_matrix_mll, covariance_matrix_map = \
    collect_reconstruction_data(
        path_to_data,
        i
        )
    LL_mll, LL_map = collect_test_log_lik(path_to_data, i)
    s_mll, v_mll, lev_score_mll = extract_eigenvectors(
        covariance_matrix_mll, 
        dic['hist']['num_k_retained']
        )
    s_map, v_map, lev_score_map = extract_eigenvectors(
        covariance_matrix_map,
        dic['hist']['num_k_retained']
        )

    images_dir = os.path.join('./', name_dir, 'images')

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    create_eigs_uncertainty_subplots(
        abs_error, 
        v_mll, 
        v_map, 
        images_dir + '/eigs_plot_{}'.format(i)
        )
    create_images_fig_subplots( 
        (image, filtbackproj, recon, abs_error, std_pred_mll, std_pred_map), 
        (LL_mll, LL_map), 
        images_dir + '/main_{}'.format(i), 
        ['$x$', 'FBP (i.e.\ $A^\dagger y_{\delta})$', 
            '$x^*$', '$|x - x^*|$', 'std -- MLL', 'std -- Type-II MAP']
        )
    create_hist_plot( 
        (abs_error, std_pred_mll, std_pred_map), 
        images_dir + '/histogram_{}'.format(i), 
        ['$|x - x^*|$', 'std -- MLL', 'std -- Type-II MAP'], 
        'marginal variances'
        )
    create_hist_plot( 
        (abs_error, s_mll, s_map), 
        images_dir + '/histogram_eigs_{}'.format(i), 
        ['$|x - x^*|$', '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$ -- MLL',
            '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$ -- Type-II MAP'], 
        'eigenvalues'
        )
