import os
import numpy as np 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from warnings import warn
try:
    import tensorflow as tf
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    from tensorflow.errors import DataLossError
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False
from tqdm import tqdm

DIR_PATH = '/localdata/experiments/dip_bayesian_ext'
OUT_PATH = './images_kmnist'

def get_line_kwargs(kws=None, add_highlight_kws=None):
    kws = kws or {}
    kws.setdefault('linewidth', 0.85)
    kws.setdefault('alpha', 0.75)
    highlight_kws = kws.copy()
    highlight_kws.update(add_highlight_kws or {})
    kws.setdefault('color', 'gray')
    highlight_kws.setdefault('color', 'red')
    return kws, highlight_kws

def tensorboard_fig_subplots(data, highlight_idx, kws=None, lin_kws=None, add_highlight_kws=None, lin_add_highlight_kws=None):

    dip_list, lin_list = data

    fig, axs = plt.subplots(1, 5, figsize=(8, 1.6),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = min(len(dip_list[0]['gp_lengthscale_0_steps']), len(lin_list[0]['gp_lengthscale_0_steps']))
    lin_kws = (lin_kws or {}).copy()
    lin_kws.setdefault('linestyle', 'dashed')
    kws, dip_highlight_kws = get_line_kwargs(kws=kws, add_highlight_kws=add_highlight_kws)
    lin_kws, lin_highlight_kws = get_line_kwargs(kws=lin_kws, add_highlight_kws=lin_add_highlight_kws)
    # TODO check names, signs and scalings
    name_list = ['negative_map_mll_scalar',  'predcp_scalars', 'posterior_hess_log_det_obs_space_scalars', 'weight_prior_log_prob_scalars',  'obs_log_density_scalars']
    label_list = ['$-{\mathcal{G}}(\sigma^{2}_{y}, \\boldsymbol{\ell}, \\boldsymbol{\sigma}_{\\boldsymbol{\\theta}}^{2})$',
         '${\\rm log } p(\\boldsymbol{\ell})$', '${\\rm log} |H|$', '${\\rm log} p(\\boldsymbol{\\theta}^*)$', '${\\rm log} p(\mathbf{y}_{\delta}|\\boldsymbol{\\theta}^*)$']
    for i, ax in enumerate(axs.flatten()):
        dip_tmp = []
        lin_tmp= []
        for dip_dic, lin_dic in zip(dip_list, lin_list):
            dip_tmp.append([v for k, v in dip_dic.items() if (name_list[i] in k and 'steps' not in k)])
            lin_tmp.append([v for k, v in lin_dic.items() if (name_list[i] in k and 'steps' not in k)])
        for k, (dip, lin) in enumerate(zip(dip_tmp, lin_tmp)):
            if k == highlight_idx: 
                if name_list[i] != 'predcp_scalars':
                    ax.plot(list(range(N)), dip[0][:N], zorder=10, **dip_highlight_kws, label='DIP')
                ax.plot(list(range(N)), lin[0][:N], zorder=10, **lin_highlight_kws, label='linearized')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if name_list[i] != 'predcp_scalars':
                ax.plot(list(range(N)), dip[0][:N], zorder=0, **kws)
            ax.plot(list(range(N)), lin[0][:N], zorder=0, **lin_kws)
        ax.set_title(label_list[i], pad=7)
        ax.grid(0.25)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    lgd = axs.flatten()[0].legend(loc = 'upper right')
    for ax in axs.flat:
        ax.set_xlabel('iteration')
    return fig

def hyperparamters_fig_subplots_gp(data, highlight_idx, sort_index=None, add_to_titles=None, kws=None, lin_kws=None, add_highlight_kws=None, lin_add_highlight_kws=None):

    dip_list, lin_list = data

    num_gp_priors = 5
    fig, axs = plt.subplots(2, num_gp_priors, figsize=(8, 3),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = min(len(dip_list[0]['gp_lengthscale_0_steps']), len(lin_list[0]['gp_lengthscale_0_steps']))
    lin_kws = (lin_kws or {}).copy()
    lin_kws.setdefault('linestyle', 'dashed')
    kws, dip_highlight_kws = get_line_kwargs(kws=kws, add_highlight_kws=add_highlight_kws)
    lin_kws, lin_highlight_kws = get_line_kwargs(kws=lin_kws, add_highlight_kws=lin_add_highlight_kws)
    sort_index = sort_index or range(num_gp_priors)
    add_to_titles = add_to_titles or [None] * num_gp_priors
    for i, ax in enumerate(axs.flatten()):
        dip_lengthscales = []
        lin_lengthscales = []
        if i < num_gp_priors: 
            for dip_dic, lin_dic in zip(dip_list, lin_list):
                dip_lengthscales.append([v for k, v in dip_dic.items() if ('lengthscale_{}'.format(sort_index[i]) in k and 'steps' not in k)])
                lin_lengthscales.append([v for k, v in lin_dic.items() if ('lengthscale_{}'.format(sort_index[i]) in k and 'steps' not in k)])
            for k, (dip_lengthscale, lin_lengthscale) in enumerate(zip(dip_lengthscales, lin_lengthscales)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), dip_lengthscale[0][:N], zorder=10, **dip_highlight_kws, label='DIP')
                    ax.plot(list(range(N)), lin_lengthscale[0][:N], zorder=10, **lin_highlight_kws, label='linearized')
                ax.plot(list(range(N)), dip_lengthscale[0][:N], zorder=0, **kws)
                ax.plot(list(range(N)), lin_lengthscale[0][:N], zorder=0, **lin_kws)
            ax.set_title('$\ell_{}$'.format(i) +
                         ('' if not add_to_titles[i] else ' --- {}'.format(add_to_titles[i])), pad=5)
            ax.grid(0.25)
        dip_variances = []
        lin_variances = []
        if i >= num_gp_priors:
            for dip_dic, lin_dic in zip(dip_list, lin_list):
                dip_variances.append([v for k, v in dip_dic.items() if ('variance_{}'.format(sort_index[i-num_gp_priors]) in k and 'steps' not in k)])
                lin_variances.append([v for k, v in lin_dic.items() if ('variance_{}'.format(sort_index[i-num_gp_priors]) in k and 'steps' not in k)])
            for k, (dip_variance, lin_variance) in enumerate(zip(dip_variances, lin_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), dip_variance[0][:N], zorder=10, **dip_highlight_kws)
                    ax.plot(list(range(N)), lin_variance[0][:N], zorder=10, **lin_highlight_kws)
                ax.plot(list(range(N)), dip_variance[0][:N], zorder=0, **kws)
                ax.plot(list(range(N)), lin_variance[0][:N], zorder=0, **lin_kws)   
            ax.set_title('$\\sigma^{%d}_{\\boldsymbol{\\theta}, %d}$' % (2, i - num_gp_priors) +
                         ('' if not add_to_titles[i - num_gp_priors] else ' --- {}'.format(add_to_titles[i - num_gp_priors])), pad=7)
            ax.grid(0.25)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, steps=[1,2,5,10]))
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([' ', ' ', ' ', ' '])
    for ax in axs[-1, :].flat:
        ax.set_xlabel('iteration')
    axs[0, 0].legend(loc = 'upper right')
    return fig

def hyperparamters_fig_subplots_normal_and_noise(data, highlight_idx, sort_index=None, add_to_titles=None, kws=None, lin_kws=None, add_highlight_kws=None, lin_add_highlight_kws=None):

    dip_list, lin_list = data

    num_normal_priors = 3
    fig, axs = plt.subplots(1, 4, figsize=(6, 1.54),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = min(len(dip_list[0]['normal_variance_0_steps']), len(lin_list[0]['normal_variance_0_steps']))
    lin_kws = (lin_kws or {}).copy()
    lin_kws.setdefault('linestyle', 'dashed')
    kws, dip_highlight_kws = get_line_kwargs(kws=kws, add_highlight_kws=add_highlight_kws)
    lin_kws, lin_highlight_kws = get_line_kwargs(kws=lin_kws, add_highlight_kws=lin_add_highlight_kws)
    sort_index = sort_index or range(num_normal_priors)
    add_to_titles = add_to_titles or [None] * num_normal_priors
    for i, ax in enumerate(axs.flatten()):
        if i < num_normal_priors:
            dip_normal_variances = []
            lin_normal_variances = []
            for dip_dic, lin_dic in zip(dip_list, lin_list):
                dip_normal_variances.append([v for k, v in dip_dic.items() if ('normal_variance_{}'.format(sort_index[i]) in k and 'steps' not in k)])
                lin_normal_variances.append([v for k, v in lin_dic.items() if ('normal_variance_{}'.format(sort_index[i]) in k and 'steps' not in k)])
            for k, (dip_normal_variance, lin_normal_variance) in enumerate(zip(dip_normal_variances, lin_normal_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), dip_normal_variance[0][:N], zorder=10, **dip_highlight_kws, label='DIP')
                    ax.plot(list(range(N)), lin_normal_variance[0][:N], zorder=10, **lin_highlight_kws, label='linearized')
                ax.plot(list(range(N)), dip_normal_variance[0][:N], zorder=0, **kws)
                ax.plot(list(range(N)), lin_normal_variance[0][:N], zorder=0, **lin_kws)
            ax.set_title('$\\sigma^{%d}_{1\\times 1,\\boldsymbol{\\theta}, %d}$' % (2, i) +
                         ('' if not add_to_titles[i] else ' --- {}'.format(add_to_titles[i])), pad=7)
            ax.grid(0.25)
        elif i == num_normal_priors:
            dip_noise_model_variances = []
            lin_noise_model_variances = []
            for dip_dic, lin_dic in zip(dip_list, lin_list):
                dip_noise_model_variances.append([v for k, v in dip_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
                lin_noise_model_variances.append([v for k, v in lin_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
            for k, (dip_noise_model_variance, lin_noise_model_variance) in enumerate(zip(dip_noise_model_variances, lin_noise_model_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), dip_noise_model_variance[0][:N], zorder=10, **dip_highlight_kws, label='DIP')
                    ax.plot(list(range(N)), lin_noise_model_variance[0][:N], zorder=10, **lin_highlight_kws, label='linearized')  
                ax.plot(list(range(N)), dip_noise_model_variance[0][:N], zorder=0, **kws)
                ax.plot(list(range(N)), lin_noise_model_variance[0][:N], zorder=0, **lin_kws)   
            ax.set_title('$\\sigma^{%d}_y$' % 2, pad=7)
            ax.grid(0.25)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, steps=[1,2,5,10]))
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[3].legend(loc = 'upper right')
    for ax in axs.flat:
        ax.set_xlabel('iteration')
    return fig

def extract_tensorboard_scalars(log_file=None, save_as_npz=None, tags=None):
    assert TF_AVAILABLE

    def my_summary_iterator(path):
        try:
            for r in tf_record.tf_record_iterator(path):
                yield event_pb2.Event.FromString(r)
        except DataLossError:
            warn('DataLossError occured, terminated reading file')

    if tags is not None:
        tags = [t.replace('/', '_').lower() for t in tags]
    values = {}
    try:
        for event in tqdm(my_summary_iterator(log_file)):
            if event.WhichOneof('what') != 'summary':
                continue
            step = event.step
            for value in event.summary.value:
                use_value = True
                if hasattr(value, 'simple_value'):
                    v = value.simple_value
                elif value.tensor.ByteSize():
                    v = tf.make_ndarray(value.tensor)
                else:
                    use_value = False
                if use_value:
                    tag = value.tag.replace('/', '_').lower()
                    if tags is None or tag in tags:
                        values.setdefault(tag, []).append((step, v))
    except DataLossError as e:
        warn('stopping for log_file "{}" due to DataLossError: {}'.format(
            log_file, e))
    scalars = {}
    for k in values.keys():
        v = np.asarray(values[k])
        steps, steps_counts = np.unique(v[:, 0], return_counts=True)
        scalars[k + '_steps'] = steps
        scalars[k + '_scalars'] = v[np.cumsum(steps_counts)-1, 1]

    if save_as_npz is not None:
        np.savez(save_as_npz, **scalars)

    return scalars

def find_tensorboard_paths(path, num_samples=50):
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_no_predcp_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
        assert len(path_to_tb) == 1
        paths_to_tb.append(path_to_tb[0])
    return paths_to_tb

def load_or_convert_log_file(log_file, npz_cache_filepath=None):
    if npz_cache_filepath is None:
        npz_cache_filepath = log_file + '.npz'
    log = None
    if os.path.isfile(npz_cache_filepath):
        log = np.load(npz_cache_filepath)
    if log is None:
        os.makedirs(os.path.dirname(npz_cache_filepath), exist_ok=True)
        extract_tensorboard_scalars(log_file=log_file, save_as_npz=npz_cache_filepath)
        log = np.load(npz_cache_filepath)

    return log

def plot():

    fs_m1 = 6  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 12  #  figure titles

    color = '#5555ff'
    color_lin = '#d2642d'

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

    angles = 20
    noise = 0.05

    highlight_idx = 0

    run_path = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-25T18:39:01.923124Z'
    run_path_lin = '/localdata/experiments/dip_bayesian_ext/outputs/2022-01-25T18:39:01.924045Z'
    run_path = os.path.join(DIR_PATH, 'outputs', run_path.split('/outputs/')[-1])  # translate to local path
    run_path_lin = os.path.join(DIR_PATH, 'outputs', run_path_lin.split('/outputs/')[-1])  # translate to local path

    setting = 'angles_{}_noise_{}'.format(angles, int(noise*100))
    exp_name = 'calibration_uncertainty_estimates'

    num_samples = 10

    tb_paths = find_tensorboard_paths(run_path, num_samples=num_samples)
    lin_tb_paths = find_tensorboard_paths(run_path_lin, num_samples=num_samples)

    logs = [load_or_convert_log_file(p) for p in tb_paths]
    lin_logs = [load_or_convert_log_file(p) for p in lin_tb_paths]

    images_dir = OUT_PATH

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    kwargs = {
        'highlight_idx': highlight_idx,
        'kws': {'color': color, 'alpha': 0.25, 'linewidth': 0.7},
        'add_highlight_kws': {'alpha': 1., 'linewidth': 1.5},
        'lin_kws': {'color': color_lin, 'alpha': 0.25, 'linewidth': 0.7},
        'lin_add_highlight_kws': {'alpha': 1., 'linewidth': 1.5},
    }

    fig = tensorboard_fig_subplots((logs, lin_logs), **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_optim_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_optim_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_subplots_gp((logs, lin_logs), sort_index=[4, 0, 1, 2, 3], add_to_titles=['\\texttt{In}', '\\texttt{Down}', '\\texttt{Down}', '\\texttt{Up}', '\\texttt{Up}'], **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_hyperparams_gp_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_hyperparams_gp_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_subplots_normal_and_noise((logs, lin_logs), add_to_titles=['\\texttt{Skip}', '\\texttt{Skip}', '\\texttt{Out}'], **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_hyperparams_normal_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_lin_weights_exp_hyperparams_normal_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()

if __name__ == "__main__": 
    plot()
