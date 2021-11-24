import os
import numpy as np 
import bios
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

DIR_PATH='/media/chen/Res/dip_bayesian_ext/src/experiments' 

run_paths = bios.read(os.path.join(DIR_PATH, 'evaluation/runs.yaml'))
name = 'kmnist_sparse_10' # kmnist_sparse_20
idx = 0 # [0, 1]
exp_name = 'calibration_uncertainty_estimates'

dic = {'images':
        {   'num': 10, 
            'n_rows': 2,
            'n_cols': 4,
            'figsize': (7.25, 3.5),
            'idx_to_norm': [4, 6],
            'idx_add_cbar': [2, 6],
            'idx_add_psnr': [1, 2],
            'idx_add_min_max': [4, 5, 6], 
            'idx_add_test_loglik': [5, 6],
            'idx_hist_insert': [3, 7],
            'idx_remove_ticks': [3],
            'text_insert':
            {
                'psnr': [5.75, 29], 
                'min_max': [-2, 14.75], 
                'test_loglik': [12, 29], 
                'fontsize': 7.5,
            }
        },
    'hist': {
            'num_bins': 25,
            'num_k_retained': 5, 
            'opacity': [0.3, 0.3, 0.3], 
            'zorder': [10, 5, 0],
            'color': ['#e63946', '#ee9b00', '#606c38'], 
            'linewidth': 0.75, 
            },
    'eigenUQ': 
    {
        'n_rows': 2,
        'n_cols': 5,
            
    }
}

def create_eigs_uncertainty_subplots(v_mll, v_map, filename):

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

def main_image_fig_subplots(data, loglik, filename, titles):

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
    vmax = np.max(data[dic['images']['idx_to_norm'][0]:dic['images']['idx_to_norm'][1]])
    vmin = np.min(data[dic['images']['idx_to_norm'][0]:dic['images']['idx_to_norm'][1]])
    k = 0 
    for i, (el, ax, title) in enumerate(zip(data, axs.flatten(), titles)):
        if i in dic['images']['idx_hist_insert']:
            kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)
            for (el, alpha, zorder, color, label) in zip(el[0], dic['hist']['opacity'], 
                dic['hist']['zorder'], dic['hist']['color'], el[1]):
                    ax.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                        facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
            ax.set_title(title, y=1.01)
            ax.set_xlim([0, 0.65])
            ax.set_ylim([0.09, 90])
            ax.set_yscale('log')
            ax.legend()
            ax.grid(alpha=0.3)
            if i in dic['images']['idx_remove_ticks']: 
                ax.set_xticklabels([' ', ' ', ' ', ' '])
        else:
            im = ax.imshow(el, cmap='gray', vmin=0, vmax=1) if i < dic['images']['idx_to_norm'][0] \
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

def create_hist_plot(data, filename, label_list, title, remove_ticks=False):

    fs_m1 = 6  # for figure ticks
    fs = 8  # for regular figure text
    fs_p1 = 9  #  figure titles

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

    fig, axs = plt.subplots(1, 1, figsize=(2,2),  facecolor='w', 
        edgecolor='k', constrained_layout=True)

    kws = dict(histtype= "stepfilled", linewidth = dic['hist']['linewidth'], ls='dashed', density=True)
    for (el, alpha, zorder, color, label) in zip(data, dic['hist']['opacity'], 
        dic['hist']['zorder'], dic['hist']['color'], label_list):
            axs.hist(el.flatten(), bins=dic['hist']['num_bins'], zorder=zorder,
                 facecolor=hex_to_rgb(color, alpha), edgecolor=hex_to_rgb(color, alpha=1), label=label, **kws)
    axs.set_title(title, y=1.01)
    axs.set_xlim([0, 0.65])
    axs.set_ylim([0.09, 90])
    axs.grid(alpha=0.3)
    axs.legend()
    axs.set_yscale('log')
    if remove_ticks: 
        axs.set_xticklabels([' ', ' ', ' ', ' '])
    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


def collect_test_log_lik(path, idx):
  
    test_log_lik_info = np.load(os.path.join(DIR_PATH, path, \
        'test_log_lik_info_{}.npz'.format(idx)), allow_pickle=True)
    return (test_log_lik_info['test_log_lik'].item()['MLL'], \
        test_log_lik_info['test_log_lik'].item()['type-II-MAP'] )

def collect_reconstruction_data(path, idx):
    
    data = np.load(os.path.join(DIR_PATH, path, 'recon_info_{}.npz'.format(idx)))

    image = \
        data['image'].squeeze()
    filtbackproj = \
        data['filtbackproj'].squeeze()
    recon = \
        data['recon'].squeeze()
    abs_error = \
        np.abs(image - recon).squeeze()
    
    try: 
        model_cov_matrix_mll = \
            data['model_post_cov_no_predCP']
    except: 
        model_cov_matrix_mll = \
            data['model_post_cov_no_PredCP']

    model_cov_matrix_map = \
        data['model_post_cov_w_PredCP']
    std_pred_mll = \
        np.sqrt(np.diag(model_cov_matrix_mll)).reshape(28, 28)
    std_pred_map = \
        np.sqrt(np.diag(model_cov_matrix_map)).reshape(28, 28)
    try: 
        Kxx_mll = data['Kxx_no_PredCP']
        Kxx_map = data['Kxx_w_PredCP']
    except:
        Kxx_mll = None 
        Kxx_map = None 

    return (image, filtbackproj, recon, abs_error, std_pred_mll, std_pred_map,
                model_cov_matrix_mll, model_cov_matrix_map, Kxx_mll, Kxx_map)
 
def extract_eigenvectors(data, k):

    s, v = np.linalg.eigh(data)
    lev_score = ( (v[:, -10:])**2 ).sum(axis=1) 
    lev_score = ( lev_score - np.min(lev_score) ) / ( np.max(lev_score) - np.min(lev_score) )
    return np.sqrt(s), v[:, -k:], lev_score.reshape(28, 28)

if __name__ == "__main__": 

    for i in range(dic['images']['num']):

        name_dir, path_to_data = run_paths[exp_name][name][idx]['name'], \
            run_paths[exp_name][name][idx]['path']
        (
        image, filtbackproj, recon, abs_error, std_pred_mll, std_pred_map,
            covariance_matrix_mll, covariance_matrix_map, Kxx_mll, Kxx_map
        ) = \
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

        # create_eigs_uncertainty_subplots(
        #     v_mll, 
        #     v_map, 
        #     images_dir + '/eigs_plot_{}'.format(i)
        #     )

        main_image_fig_subplots( 
            (
            image, filtbackproj, recon, 
                (
                    (abs_error, std_pred_mll, std_pred_map),  
                    ['$|{\mathbf{x} - \mathbf{x}^*}|$', 'std-dev (MLL)', 'std-dev  (Type-II MAP)']
                ),
            abs_error, std_pred_mll, std_pred_map, 
            (
                (abs_error, s_mll, s_map),
                ['$|{\mathbf{x} - \mathbf{x}^*}|$', '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$ (MLL)',
                '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$  (Type-II MAP)'])
            ),
            (LL_mll, LL_map),
            images_dir + '/main_{}'.format(i), 
            ['${\mathbf{x}}$', 'FBP (i.e.\ ${\\textnormal{A}}^\dagger y_{\delta})$', 
            '${\mathbf{x}^*}$', 'marginal std-dev', '${|\mathbf{x} - \mathbf{x}^*|}$', 
            'std-dev  (MLL)', 'std-dev (Type-II MAP)', 'covariance eigenspectrum']
            )

        # create_hist_plot( 
        #     (abs_error, std_pred_mll, std_pred_map), 
        #     images_dir + '/histogram_{}'.format(i), 
        #     ['$|x - x^*|$', 'std -- MLL', 'std -- Type-II MAP'], 
        #     'marginal std',
        #     True
        #     )

        # create_hist_plot( 
        #     (abs_error, s_mll, s_map), 
        #     images_dir + '/histogram_eigs_{}'.format(i), 
        #     ['$|x - x^*|$', '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$ -- MLL',
        #         '$\lambda(\Sigma_{\mathbf{f}|\mathbf{y}_{\delta}})^{0.5}$ -- Type-II MAP'], 
        #     'eigenspectrum', 
        #     False
        #     )
