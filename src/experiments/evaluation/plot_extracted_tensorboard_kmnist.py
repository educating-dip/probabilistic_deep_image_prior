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

RUN_FILE = '/localdata/jleuschn/experiments/dip_bayesian_ext/kmnist_refined_tv_strength.yaml'
DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext'
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

def tensorboard_fig_subplots(data, highlight_idx, mll_kws=None, map_kws=None, mll_add_highlight_kws=None, map_add_highlight_kws=None):

    mll_list, map_list = data

    fig, axs = plt.subplots(1, 5, figsize=(8, 1.6),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(mll_list[0]['gp_lengthscale_0_steps'])
    map_kws = (map_kws or {}).copy()
    map_kws.setdefault('linestyle', 'dashed')
    mll_kws, mll_highlight_kws = get_line_kwargs(kws=mll_kws, add_highlight_kws=mll_add_highlight_kws)
    map_kws, map_highlight_kws = get_line_kwargs(kws=map_kws, add_highlight_kws=map_add_highlight_kws)
    # TODO check names, signs and scalings
    name_list = ['negative_map_mll_scalar',  'predcp_scalars', 'posterior_hess_log_det_obs_space_scalars', 'weight_prior_log_prob_scalars',  'obs_log_density_scalars']
    label_list = ['$-{\mathcal{G}}(\sigma^{2}_{y}, \\boldsymbol{\ell}, \\boldsymbol{\sigma}_{\\boldsymbol{\\theta}}^{2})$',
         '${\\rm log } p(\\boldsymbol{\ell})$', '${\\rm log} |H|$', '${\\rm log} p(\\boldsymbol{\\theta}^*)$', '${\\rm log} p(\mathbf{y}_{\delta}|\\boldsymbol{\\theta}^*)$']
    for i, ax in enumerate(axs.flatten()):
        mll_tmp = []
        map_tmp= []
        for mll_dic, map_dic in zip(mll_list, map_list):
            mll_tmp.append([v for k, v in mll_dic.items() if (name_list[i] in k and 'steps' not in k)])
            map_tmp.append([v for k, v in map_dic.items() if (name_list[i] in k and 'steps' not in k)])
        for k, (mll, map) in enumerate(zip(mll_tmp, map_tmp)):
            if k == highlight_idx: 
                if name_list[i] != 'predcp_scalars':
                    ax.plot(list(range(N)), mll[0], zorder=10, **mll_highlight_kws, label='MLL')
                ax.plot(list(range(N)), map[0], zorder=10, **map_highlight_kws, label='TV-MAP')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            if name_list[i] != 'predcp_scalars':
                ax.plot(list(range(N)), mll[0], zorder=0, **mll_kws)
            ax.plot(list(range(N)), map[0], zorder=0, **map_kws)
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

def hyperparamters_fig_subplots_gp(data, highlight_idx, sort_index=None, add_to_titles=None, mll_kws=None, map_kws=None, mll_add_highlight_kws=None, map_add_highlight_kws=None):

    mll_list, map_list = data

    num_gp_priors = 5
    fig, axs = plt.subplots(2, num_gp_priors, figsize=(8, 3),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(mll_list[0]['gp_lengthscale_0_steps'])
    map_kws = (map_kws or {}).copy()
    map_kws.setdefault('linestyle', 'dashed')
    mll_kws, mll_highlight_kws = get_line_kwargs(kws=mll_kws, add_highlight_kws=mll_add_highlight_kws)
    map_kws, map_highlight_kws = get_line_kwargs(kws=map_kws, add_highlight_kws=map_add_highlight_kws)
    sort_index = sort_index or range(num_gp_priors)
    add_to_titles = add_to_titles or [None] * num_gp_priors
    for i, ax in enumerate(axs.flatten()):
        mll_lengthscales = []
        map_lengthscales = []
        if i < num_gp_priors: 
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_lengthscales.append([v for k, v in mll_dic.items() if ('lengthscale_{}'.format(sort_index[i]) in k and 'steps' not in k)])
                map_lengthscales.append([v for k, v in map_dic.items() if ('lengthscale_{}'.format(sort_index[i]) in k and 'steps' not in k)])
            for k, (mll_lengthscale, map_lengthscale) in enumerate(zip(mll_lengthscales, map_lengthscales)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), mll_lengthscale[0], zorder=10, **mll_highlight_kws, label='MLL')
                    ax.plot(list(range(N)), map_lengthscale[0], zorder=10, **map_highlight_kws, label='TV-MAP')
                ax.plot(list(range(N)), mll_lengthscale[0], zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_lengthscale[0], zorder=0, **map_kws)
            ax.set_title('$\ell_{}$'.format(i) +
                         ('' if not add_to_titles[i] else ' --- {}'.format(add_to_titles[i])), pad=5)
            ax.grid(0.25)
        mll_variances = []
        map_variances = []
        if i >= num_gp_priors:
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_variances.append([v for k, v in mll_dic.items() if ('variance_{}'.format(sort_index[i-num_gp_priors]) in k and 'steps' not in k)])
                map_variances.append([v for k, v in map_dic.items() if ('variance_{}'.format(sort_index[i-num_gp_priors]) in k and 'steps' not in k)])
            for k, (mll_variance, map_variance) in enumerate(zip(mll_variances, map_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), mll_variance[0], zorder=10, **mll_highlight_kws)
                    ax.plot(list(range(N)), map_variance[0], zorder=10, **map_highlight_kws)
                ax.plot(list(range(N)), mll_variance[0], zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_variance[0], zorder=0, **map_kws)   
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
    axs[0, 1].legend(loc = 'upper right')
    return fig

def hyperparamters_fig_subplots_normal_and_noise(data, highlight_idx, sort_index=None, add_to_titles=None, mll_kws=None, map_kws=None, mll_add_highlight_kws=None, map_add_highlight_kws=None):

    mll_list, map_list = data

    num_normal_priors = 3
    fig, axs = plt.subplots(1, 4, figsize=(6, 1.54),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(mll_list[0]['normal_variance_0_steps'])
    map_kws = (map_kws or {}).copy()
    map_kws.setdefault('linestyle', 'dashed')
    mll_kws, mll_highlight_kws = get_line_kwargs(kws=mll_kws, add_highlight_kws=mll_add_highlight_kws)
    map_kws, map_highlight_kws = get_line_kwargs(kws=map_kws, add_highlight_kws=map_add_highlight_kws)
    sort_index = sort_index or range(num_normal_priors)
    add_to_titles = add_to_titles or [None] * num_normal_priors
    for i, ax in enumerate(axs.flatten()):
        if i < num_normal_priors:
            mll_normal_variances = []
            map_normal_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_normal_variances.append([v for k, v in mll_dic.items() if ('normal_variance_{}'.format(sort_index[i]) in k and 'steps' not in k)])
                map_normal_variances.append([v for k, v in map_dic.items() if ('normal_variance_{}'.format(sort_index[i]) in k and 'steps' not in k)])
            for k, (mll_normal_variance, map_normal_variance) in enumerate(zip(mll_normal_variances, map_normal_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), mll_normal_variance[0], zorder=10, **mll_highlight_kws, label='MLL')
                    ax.plot(list(range(N)), map_normal_variance[0], zorder=10, **map_highlight_kws, label='TV-MAP')
                ax.plot(list(range(N)), mll_normal_variance[0], zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_normal_variance[0], zorder=0, **map_kws)
            ax.set_title('$\\sigma^{%d}_{1\\times 1,\\boldsymbol{\\theta}, %d}$' % (2, i) +
                         ('' if not add_to_titles[i] else ' --- {}'.format(add_to_titles[i])), pad=7)
            ax.grid(0.25)
        elif i == num_normal_priors:
            mll_noise_model_variances = []
            map_noise_model_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_noise_model_variances.append([v for k, v in mll_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
                map_noise_model_variances.append([v for k, v in map_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
            for k, (mll_noise_model_variance, map_noise_model_variance) in enumerate(zip(mll_noise_model_variances, map_noise_model_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), mll_noise_model_variance[0][:N], zorder=10, **mll_highlight_kws, label='MLL')
                    ax.plot(list(range(N)), map_noise_model_variance[0][:N], zorder=10, **map_highlight_kws, label='TV-MAP')  
                ax.plot(list(range(N)), mll_noise_model_variance[0][:N], zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_noise_model_variance[0][:N], zorder=0, **map_kws)   
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

def find_tensorboard_paths_mll(path, num_samples=50):
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_no_predcp_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
        assert len(path_to_tb) == 1
        paths_to_tb.append(path_to_tb[0])
    return paths_to_tb

def find_tensorboard_paths_map(path, num_samples=50):
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_predcp_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
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

    color_mll = '#5555ff'
    color_map = '#5a6c17'

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

    highlight_idx = 46

    with open(RUN_FILE, 'r') as f:
        runs_dict = yaml.safe_load(f)

    run_path = runs_dict[angles][noise]
    run_path = os.path.join(DIR_PATH, 'outputs', run_path.split('/outputs/')[-1])  # translate to local path

    setting = 'angles_{}_noise_{}'.format(angles, int(noise*100))
    exp_name = 'calibration_uncertainty_estimates'

    num_samples = 50

    mll_tb_paths = find_tensorboard_paths_mll(run_path, num_samples=num_samples)
    map_tb_paths = find_tensorboard_paths_map(run_path, num_samples=num_samples)

    mll_logs = [load_or_convert_log_file(p) for p in mll_tb_paths]
    map_logs = [load_or_convert_log_file(p) for p in map_tb_paths]

    images_dir = OUT_PATH

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    kwargs = {
        'highlight_idx': highlight_idx,
        'mll_kws': {'color': color_mll, 'alpha': 0.25, 'linewidth': 0.7},
        'mll_add_highlight_kws': {'alpha': 1., 'linewidth': 1.5},
        'map_kws': {'color': color_map, 'alpha': 0.25, 'linewidth': 0.7},
        'map_add_highlight_kws': {'alpha': 1., 'linewidth': 1.5},
    }

    fig = tensorboard_fig_subplots((mll_logs, map_logs), **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_optim_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_optim_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_subplots_gp((mll_logs, map_logs), sort_index=[4, 0, 1, 2, 3], add_to_titles=['\\texttt{In}', '\\texttt{Down}', '\\texttt{Down}', '\\texttt{Up}', '\\texttt{Up}'], **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_hyperparams_gp_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_hyperparams_gp_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_subplots_normal_and_noise((mll_logs, map_logs), add_to_titles=['\\texttt{Skip}', '\\texttt{Skip}', '\\texttt{Out}'], **kwargs)
    fig.savefig(os.path.join(images_dir, 'kmnist_hyperparams_normal_{}_{}.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_hyperparams_normal_{}_{}.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()

if __name__ == "__main__": 
    plot()
