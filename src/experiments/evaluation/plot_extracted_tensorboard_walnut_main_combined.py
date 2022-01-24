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
from omegaconf import OmegaConf

DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext/'
OUT_PATH = './images_walnut'

run_path_mll_no_clamping = 'outputs/2022-01-13T00:14:58.653648Z'
run_path_map_no_clamping = 'outputs/2022-01-13T00:14:58.653639Z'
run_path_mll = 'outputs/2022-01-15T17:02:21.405935Z'
run_path_map = 'outputs/2022-01-15T17:02:21.406508Z'

def get_line_kwargs(kws=None):
    kws = kws or {}
    kws.setdefault('linewidth', 0.85)
    kws.setdefault('alpha', 0.75)
    kws.setdefault('color', 'gray')
    return kws

def hyperparamters_fig_subplots_main(data, iterations=None, mll_kws=None, map_kws=None, empirical_noise_variance=None):

    mll_list, map_list = data

    num_normal_priors = 3
    fig, axs = plt.subplots(2, 2, figsize=(3, 2.75),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = iterations if iterations is not None else len(mll_list[0]['normal_variance_0_steps'])
    map_kws = (map_kws or {}).copy()
    map_kws.setdefault('linestyle', 'dashed')
    mll_kws = get_line_kwargs(kws=mll_kws)
    map_kws = get_line_kwargs(kws=map_kws)
    sort_index_gp = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    add_to_titles_gp = ['\\texttt{In}'] + ['\\texttt{Down}'] * 5 + ['\\texttt{Up}'] * 5
    sort_index_normal = range(num_normal_priors)
    add_to_titles_normal = ['\\texttt{Skip}', '\\texttt{Skip}', '\\texttt{Out}']
    for i, ax in enumerate(axs.flatten()):
        if i == 0:
            mll_lengthscales = []
            map_lengthscales = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_lengthscales.append([v for k, v in mll_dic.items() if ('gp_lengthscale_{}'.format(sort_index_gp[10]) in k and 'steps' not in k)])
                map_lengthscales.append([v for k, v in map_dic.items() if ('gp_lengthscale_{}'.format(sort_index_gp[10]) in k and 'steps' not in k)])
            for k, (mll_lengthscale, map_lengthscale) in enumerate(zip(mll_lengthscales, map_lengthscales)):
                h_mll = ax.plot(list(range(N)), mll_lengthscale[0][:N], zorder=10, **mll_kws, label='MLL' if k == 0 else None)
                h_map = ax.plot(list(range(N)), map_lengthscale[0][:N], zorder=10, **map_kws, label='TV-MAP' if k == 0 else None)
            ax.set_title('$\ell_{{{}}}$'.format(10) +
                        ('' if not add_to_titles_gp[10] else ' --- {}'.format(add_to_titles_gp[10])), pad=5)
            ax.grid(0.25)
        if i == 1:
            mll_variances = []
            map_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_variances.append([v for k, v in mll_dic.items() if ('gp_variance_{}'.format(sort_index_gp[10]) in k and 'steps' not in k)])
                map_variances.append([v for k, v in map_dic.items() if ('gp_variance_{}'.format(sort_index_gp[10]) in k and 'steps' not in k)])
            for k, (mll_variance, map_variance) in enumerate(zip(mll_variances, map_variances)):
                ax.plot(list(range(N)), mll_variance[0][:N], zorder=10, **mll_kws)
                ax.plot(list(range(N)), map_variance[0][:N], zorder=10, **map_kws)
            ax.set_title('$\\sigma^{%d}_{\\boldsymbol{\\theta}, %d}$' % (2, 10) +
                         ('' if not add_to_titles_gp[10] else ' --- {}'.format(add_to_titles_gp[10])), pad=7)
            ax.grid(0.25)
        if i == 2:
            mll_normal_variances = []
            map_normal_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_normal_variances.append([v for k, v in mll_dic.items() if ('normal_variance_{}'.format(sort_index_normal[2]) in k and 'steps' not in k)])
                map_normal_variances.append([v for k, v in map_dic.items() if ('normal_variance_{}'.format(sort_index_normal[2]) in k and 'steps' not in k)])
            for k, (mll_normal_variance, map_normal_variance) in enumerate(zip(mll_normal_variances, map_normal_variances)):
                ax.plot(list(range(N)), mll_normal_variance[0][:N], zorder=10, **mll_kws, label='MLL' if k == 0 else None)
                ax.plot(list(range(N)), map_normal_variance[0][:N], zorder=10, **map_kws, label='TV-MAP' if k == 0 else None)  
            ax.set_title('$\\sigma^{%d}_{1\\times 1,\\boldsymbol{\\theta}, %d}$' % (2, 2) +
                         ('' if not add_to_titles_normal[2] else ' --- {}'.format(add_to_titles_normal[2])), pad=7)
            ax.grid(0.25)
        if i == 3:
            mll_noise_model_variances = []
            map_noise_model_variances = []
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_noise_model_variances.append([v for k, v in mll_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
                map_noise_model_variances.append([v for k, v in map_dic.items() if ('noise_model_variance_obs' in k and 'steps' not in k)])
            for k, (mll_noise_model_variance, map_noise_model_variance) in enumerate(zip(mll_noise_model_variances, map_noise_model_variances)):
                ax.plot(list(range(N)), mll_noise_model_variance[0][:N], zorder=10, **mll_kws, label='MLL' if k == 0 else None)
                ax.plot(list(range(N)), map_noise_model_variance[0][:N], zorder=10, **map_kws, label='TV-MAP' if k == 0 else None)
            if empirical_noise_variance is not None:
                ax.plot([0, N-1], [empirical_noise_variance, empirical_noise_variance])[0].set_visible(False)  # only for ylim
                ax.axhline(y=empirical_noise_variance, zorder=2, color='#aa007f', linestyle=(0, (2, 1)), linewidth=1.)
            ax.set_title('$\\sigma^{%d}_y$' % 2, pad=7)
            ax.grid(0.25)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8, steps=[1,2,5,10]))
    for ax in axs[0, :].flat:
        ax.set_xticklabels([' '])
    for ax in axs[1, :].flat:
        ax.set_xlabel('iteration')
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    lgd = axs.flatten()[3].legend(loc = 'upper right')
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

def find_tensorboard_path_mll(path):
    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    assert not cfg.mrglik.optim.include_predcp
    path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_recon_num_0_*' + '/events*') if not p.endswith('.npz')]
    assert len(path_to_tb) == 1
    return path_to_tb[0]

def find_tensorboard_path_map(path):
    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
    assert cfg.mrglik.optim.include_predcp
    path_to_tb = [p for p in glob.glob(path + '/mrglik_opt_recon_num_0_*' + '/events*') if not p.endswith('.npz')]
    assert len(path_to_tb) == 1
    return path_to_tb[0]

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

    highlight_idx = 1

    full_run_path_mll_no_clamping = os.path.join(DIR_PATH, run_path_mll_no_clamping)
    full_run_path_map_no_clamping = os.path.join(DIR_PATH, run_path_map_no_clamping)
    full_run_path_mll = os.path.join(DIR_PATH, run_path_mll)
    full_run_path_map = os.path.join(DIR_PATH, run_path_map)

    setting = 'ass20_css6'
    exp_name = 'calibration_uncertainty_estimates'

    mll_tb_path_no_clamping = find_tensorboard_path_mll(full_run_path_mll_no_clamping)
    map_tb_path_no_clamping = find_tensorboard_path_map(full_run_path_map_no_clamping)
    mll_tb_path = find_tensorboard_path_mll(full_run_path_mll)
    map_tb_path = find_tensorboard_path_map(full_run_path_map)

    mll_log_no_clamping = load_or_convert_log_file(mll_tb_path_no_clamping)
    map_log_no_clamping = load_or_convert_log_file(map_tb_path_no_clamping)
    mll_log = load_or_convert_log_file(mll_tb_path)
    map_log = load_or_convert_log_file(map_tb_path)

    mll_log_combined = dict(mll_log)
    map_log_combined = dict(map_log)
    # use sigma_y from non-clamping runs
    for k in mll_log:
        if 'noise_model_variance_obs' in k:
            mll_log_combined[k] = mll_log_no_clamping[k]
            map_log_combined[k] = map_log_no_clamping[k]

    images_dir = OUT_PATH

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    kwargs = {
        'iterations': 1601,
        'mll_kws': {'color': color_mll, 'alpha': .9, 'linewidth': 1.25},
        'map_kws': {'color': color_map, 'alpha': .9, 'linewidth': 1.25},
    }

    suffix = ''

    empirical_noise_variance = 0.0352553  # for walnut 1

    fig = hyperparamters_fig_subplots_main(([mll_log_combined], [map_log_combined]), **kwargs, empirical_noise_variance=empirical_noise_variance)
    fig.savefig(os.path.join(images_dir, 'walnut_hyperparams_main_combined_{}{}.pdf'.format(setting, suffix)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'walnut_hyperparams_main_combined_{}{}.png'.format(setting, suffix)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()

if __name__ == "__main__": 
    plot()
