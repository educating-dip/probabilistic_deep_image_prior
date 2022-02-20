import os
import numpy as np 
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import glob
from omegaconf import OmegaConf
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

RUN_FILE = '/localdata/experiments/dip_bayesian_ext/kmnist_refined_tv_strength.yaml'
RUN_FILE_APPROX = '/localdata/experiments/dip_bayesian_ext/runs_kmnist_approx.yaml'
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

def hyperparamters_fig_subplots_gp(data, method_label_list, highlight_idx, sort_index=None, add_to_titles=None, kws_list=None, add_highlight_kws_list=None, legend_idx=0):

    num_gp_priors = 5
    fig, axs = plt.subplots(2, num_gp_priors, figsize=(8, 3),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(data[0][0]['gp_lengthscale_0_steps'])
    kws_list = [kws.copy() or {} for kws in (kws_list or [None] * len(data))]
    add_highlight_kws_list = [add_highlight_kws.copy() or {} for add_highlight_kws in (add_highlight_kws_list or [None] * len(data))]
    if len(data) > 1:
        kws_list[1].setdefault('linestyle', 'dashdot')
    if len(data) > 2:
        kws_list[2].setdefault('linestyle', 'dashed')
    if len(data) > 3:
        kws_list[3].setdefault('linestyle', 'dotted')
    kws_list, highlight_kws_list = map(list, zip(*[get_line_kwargs(kws, add_highlight_kws) for kws, add_highlight_kws in zip(kws_list, add_highlight_kws_list)]))
    sort_index = sort_index or range(num_gp_priors)
    add_to_titles = add_to_titles or [None] * num_gp_priors
    for i, ax in enumerate(axs.flatten()):
        if i < num_gp_priors: 
            lengthscales_list = [[] for _ in range(len(data))]
            for lengthscales, dic_list in zip(lengthscales_list, data):
                for dic in dic_list:
                    lengthscales.append([v for k, v in dic.items() if ('lengthscale_{}'.format(sort_index[i]) in k and 'steps' not in k)])
            for lengthscales, kws, highlight_kws, label in zip(lengthscales_list, kws_list, highlight_kws_list, method_label_list):
                for k, lengthscale in enumerate(lengthscales):
                    if k == highlight_idx: 
                        ax.plot(list(range(N)), lengthscale[0], zorder=10, **highlight_kws, label=label)
                    ax.plot(list(range(N)), lengthscale[0], zorder=0, **kws)
            ax.set_title('$\ell_{}$'.format(i) +
                         ('' if not add_to_titles[i] else ' --- {}'.format(add_to_titles[i])), pad=5)
            ax.grid(0.25)
        if i >= num_gp_priors:
            variances_list = [[] for _ in range(len(data))]
            for variances, dic_list in zip(variances_list, data):
                for dic in dic_list:
                    variances.append([v for k, v in dic.items() if ('variance_{}'.format(sort_index[i-num_gp_priors]) in k and 'steps' not in k)])
            for variances, kws, highlight_kws, label in zip(variances_list, kws_list, highlight_kws_list, method_label_list):
                for k, variance in enumerate(variances):
                    if k == highlight_idx: 
                        ax.plot(list(range(N)), variance[0], zorder=10, **highlight_kws, label=label)
                    ax.plot(list(range(N)), variance[0], zorder=0, **kws)
            ax.set_title('$\\sigma^{%d}_{\\boldsymbol{\\theta}, %d}$' % (2, i - num_gp_priors) +
                         ('' if not add_to_titles[i - num_gp_priors] else ' --- {}'.format(add_to_titles[i - num_gp_priors])), pad=7)
            ax.grid(0.25)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, steps=[1,2,5,10]))
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for ax in axs[:-1, :].flat:
        ax.set_xticklabels([' ', ' ', ' ', ' '])
    axs.flatten()[legend_idx].legend(loc = 'upper right')
    return fig

def hyperparamters_fig_subplots_normal_and_noise(data, highlight_idx, mll_kws=None, map_kws=None, mll_add_highlight_kws=None, map_add_highlight_kws=None):

    mll_list, map_list = data

    num_normal_priors = 3
    fig, axs = plt.subplots(1, 4, figsize=(6, 1.54),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(mll_list[0]['normal_variance_0_steps'])
    map_kws = (map_kws or {}).copy()
    map_kws.setdefault('linestyle', 'dashed')
    mll_kws, mll_highlight_kws = get_line_kwargs(kws=mll_kws, add_highlight_kws=mll_add_highlight_kws)
    map_kws, map_highlight_kws = get_line_kwargs(kws=map_kws, add_highlight_kws=map_add_highlight_kws)
    for i, ax in enumerate(axs.flatten()):
        if i < num_normal_priors:
            mll_normal_variances = []
            map_normal_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_normal_variances.append([v for k, v in mll_dic.items() if ('normal_variance_{}'.format(i) in k and 'steps' not in k)])
                map_normal_variances.append([v for k, v in map_dic.items() if ('normal_variance_{}'.format(i) in k and 'steps' not in k)])
            for k, (mll_normal_variance, map_normal_variance) in enumerate(zip(mll_normal_variances, map_normal_variances)):
                if k == highlight_idx: 
                    ax.plot(list(range(N)), mll_normal_variance[0], zorder=10, **mll_highlight_kws, label='MLL')
                    ax.plot(list(range(N)), map_normal_variance[0], zorder=10, **map_highlight_kws, label='TV-MAP')
                ax.plot(list(range(N)), mll_normal_variance[0], zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_normal_variance[0], zorder=0, **map_kws)
            ax.set_title('$\ell_{}$'.format(i), pad=5)
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

def find_tensorboard_path_mll(path, num_samples=50):
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_no_predcp_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
        assert len(path_to_tb) == 1
        paths_to_tb.append(path_to_tb[0])
    return paths_to_tb

def find_tensorboard_path_map(path, num_samples=50):
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_predcp_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
        assert len(path_to_tb) == 1
        paths_to_tb.append(path_to_tb[0])
    return paths_to_tb

def find_tensorboard_path_approx_mll(path, num_samples=50, assert_vec_batch_size=25):
    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    assert not cfg.mrglik.optim.include_predcp
    assert cfg.mrglik.impl.vec_batch_size == assert_vec_batch_size
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
        assert len(path_to_tb) == 1
        paths_to_tb.append(path_to_tb[0])
    return paths_to_tb

def find_tensorboard_path_approx_map(path, num_samples=50, assert_vec_batch_size=25):
    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    assert cfg.mrglik.optim.include_predcp
    assert cfg.mrglik.impl.vec_batch_size == assert_vec_batch_size
    paths_to_tb = []
    for i in range(num_samples):
        path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_recon_num_{}_*'.format(i) + '/events*') if not p.endswith('.npz')]
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

    highlight_idx = 0

    vec_batch_size_list = [25, 10, 5]

    with open(RUN_FILE, 'r') as f:
        runs_dict = yaml.safe_load(f)

    with open(RUN_FILE_APPROX, 'r') as f:
        runs_dict_approx = yaml.safe_load(f)

    run_path = runs_dict[angles][noise]
    run_path = os.path.join(DIR_PATH, 'outputs', run_path.split('/outputs/')[-1])  # translate to local path

    run_path_approx_mll_list = [runs_dict_approx[angles][noise][vec_batch_size]['no_predcp'] for vec_batch_size in vec_batch_size_list]
    run_path_approx_map_list = [runs_dict_approx[angles][noise][vec_batch_size]['predcp'] for vec_batch_size in vec_batch_size_list]

    setting = 'angles_{}_noise_{}'.format(angles, int(noise*100))
    exp_name = 'calibration_uncertainty_estimates'

    num_samples = 5

    tb_paths_mll = find_tensorboard_path_mll(run_path, num_samples=num_samples)
    tb_paths_map = find_tensorboard_path_map(run_path, num_samples=num_samples)
    tb_paths_approx_mll_list = [find_tensorboard_path_approx_mll(run_path_approx_mll, num_samples=num_samples, assert_vec_batch_size=vec_batch_size) for vec_batch_size, run_path_approx_mll in zip(vec_batch_size_list, run_path_approx_mll_list)]
    tb_paths_approx_map_list = [find_tensorboard_path_approx_map(run_path_approx_map, num_samples=num_samples, assert_vec_batch_size=vec_batch_size) for vec_batch_size, run_path_approx_map in zip(vec_batch_size_list, run_path_approx_map_list)]

    logs_mll = [load_or_convert_log_file(p) for p in tb_paths_mll]
    logs_map = [load_or_convert_log_file(p) for p in tb_paths_map]
    logs_approx_mll_list = [[load_or_convert_log_file(p) for p in tb_paths_approx_mll]
                            for tb_paths_approx_mll in tb_paths_approx_mll_list]
    logs_approx_map_list = [[load_or_convert_log_file(p) for p in tb_paths_approx_map]
                            for tb_paths_approx_map in tb_paths_approx_map_list]

    # exact and approx combined
    logs_list_mll = [logs_mll] + logs_approx_mll_list
    logs_list_map = [logs_map] + logs_approx_map_list

    method_label_list_mll = ['MLL exact'] + ['MLL approx.\ \#$\mathbf{{v}}$={}'.format(vec_batch_size) for vec_batch_size in vec_batch_size_list]
    method_label_list_map = ['TV-MAP exact'] + ['TV-MAP approx.\ \#$\mathbf{{v}}$={}'.format(vec_batch_size) for vec_batch_size in vec_batch_size_list]

    images_dir = OUT_PATH

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    color_mll_approx_list = ['#9c55ff', '#c655ff', '#f155ff']
    color_map_approx_list = ['#6c6817', '#6c5217', '#6c3d17']

    kwargs_mll = {
        'highlight_idx': highlight_idx,
        'kws_list': (
                [{'color': color_mll, 'alpha': 0.25, 'linewidth': 0.7}] +
                [{'color': color_mll_approx, 'alpha': 0.25, 'linewidth': 0.7} for color_mll_approx in color_mll_approx_list]),
        'add_highlight_kws_list': (
                [{'alpha': 1., 'linewidth': 1.5}] +
                [{'alpha': 1., 'linewidth': 1.5} for _ in range(len(vec_batch_size_list))]),
    }
    kwargs_map = {
        'highlight_idx': highlight_idx,
        'kws_list': (
                [{'color': color_map, 'alpha': 0.25, 'linewidth': 0.7}] +
                [{'color': color_map_approx, 'alpha': 0.25, 'linewidth': 0.7} for color_map_approx in color_map_approx_list]),
        'add_highlight_kws_list': (
                [{'alpha': 1., 'linewidth': 1.5}] +
                [{'alpha': 1., 'linewidth': 1.5} for _ in range(len(vec_batch_size_list))]),
    }

    fig = hyperparamters_fig_subplots_gp(logs_list_mll, method_label_list_mll,
            sort_index=[4, 0, 1, 2, 3], add_to_titles=['\\texttt{In}', '\\texttt{Down}', '\\texttt{Down}', '\\texttt{Up}', '\\texttt{Up}'],
            **kwargs_mll, legend_idx=4)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_gp_{}_{}_mll.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_gp_{}_{}_mll.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_subplots_gp(logs_list_map, method_label_list_map,
            sort_index=[4, 0, 1, 2, 3], add_to_titles=['\\texttt{In}', '\\texttt{Down}', '\\texttt{Down}', '\\texttt{Up}', '\\texttt{Up}'],
            **kwargs_map, legend_idx=5)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_gp_{}_{}_map.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_gp_{}_{}_map.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    # fig = hyperparamters_fig_subplots_normal_and_noise((mll_logs, map_logs), **kwargs)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_normal_{}.pdf'.format(setting)), bbox_inches='tight', pad_inches=0.)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_hyperparams_normal_{}.png'.format(setting)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()

if __name__ == "__main__": 
    plot()
