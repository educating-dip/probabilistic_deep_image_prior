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

RUN_FILE_APPROX_NO_PRECOND = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx_no_precond.yaml'
RUN_FILE_APPROX = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx.yaml'
DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext'
OUT_PATH = './images_kmnist'

# def get_line_kwargs(kws=None, add_highlight_kws=None):
#     kws = kws or {}
#     kws.setdefault('linewidth', 0.85)
#     kws.setdefault('alpha', 0.75)
#     highlight_kws = kws.copy()
#     highlight_kws.update(add_highlight_kws or {})
#     kws.setdefault('color', 'gray')
#     highlight_kws.setdefault('color', 'red')
#     return kws, highlight_kws

# def hyperparamters_fig_cg_residual(data, method_label_list, highlight_idx, kws_list=None, add_highlight_kws_list=None, highlight_zorder_list=None):

#     num_gp_priors = 5
#     fig, ax = plt.subplots(1, 1, figsize=(1.75, 1.5),
#           facecolor='w', edgecolor='k', constrained_layout=True)
#     N = len(data[0][0]['gp_lengthscale_0_steps'])
#     kws_list = [kws.copy() or {} for kws in (kws_list or [None] * len(data))]
#     add_highlight_kws_list = [add_highlight_kws.copy() or {} for add_highlight_kws in (add_highlight_kws_list or [None] * len(data))]
#     kws_list, highlight_kws_list = map(list, zip(*[get_line_kwargs(kws, add_highlight_kws) for kws, add_highlight_kws in zip(kws_list, add_highlight_kws_list)]))
#     highlight_zorder_list = highlight_zorder_list or list(range(10, 10+len(data)))
#     residuals_list = [[] for _ in range(len(data))]
#     for residuals, dic_list in zip(residuals_list, data):
#         for dic in dic_list:
#             residuals.append([v for k, v in dic.items() if ('log_det_grad_cg_mean_residual' in k and 'steps' not in k)])
#     for j, (residuals, kws, highlight_kws, label) in enumerate(zip(residuals_list, kws_list, highlight_kws_list, method_label_list)):
#         for k, residual in enumerate(residuals):
#             if k == highlight_idx: 
#                 ax.plot(list(range(N)), residual[0], zorder=highlight_zorder_list[j], **highlight_kws, label=label)
#             ax.plot(list(range(N)), residual[0], zorder=0, **kws)
#     ax.set_title('CG residual', pad=5)
#     ax.grid(0.25)
#     ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(6, steps=[1,2,5,10]))
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_xlabel('iteration')
#     ax.set_yscale('log')
#     ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.5))
#     return fig

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, line_alpha=1, ax=None, lw=1, linestyle='-', fill_linewidths=0.2):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    plt_return = ax.plot(x, y, color=color, lw=lw, linestyle=linestyle, alpha=line_alpha)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill, linewidths=fill_linewidths)
    return plt_return

def hyperparamters_fig_cg_residual(data, method_label_list, kws_list=None):

    num_gp_priors = 5
    fig, ax = plt.subplots(1, 1, figsize=(1.85, 1.5),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = len(data[0][0]['gp_lengthscale_0_steps'])
    kws_list = [kws.copy() or {} for kws in (kws_list or [{}] * len(data))]
    residuals_list = [[] for _ in range(len(data))]
    for residuals, dic_list in zip(residuals_list, data):
        for dic in dic_list:
            residuals.append([v for k, v in dic.items() if ('log_det_grad_cg_mean_residual' in k and 'steps' not in k)])
    handles_list = []
    for j, (residuals, kws, label) in enumerate(zip(residuals_list, kws_list, method_label_list)):
        mean_residuals = np.mean(residuals, axis=0)
        std_error_residuals = np.std(residuals, axis=0) / np.sqrt(len(residuals))
        h, = errorfill(list(range(N)), mean_residuals[0], std_error_residuals[0], ax=ax, **kws)
        handles_list.append(h)
    ax.set_ylabel('CG residual')
    # ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(8, steps=[1,2,5,10]))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('iteration')
    ax.set_yscale('log')
    ax.grid(which='both')
    lgd = ax.legend(handles_list, method_label_list, loc = 'upper center', bbox_to_anchor=(0.5, -0.525))
    for h in lgd.legendHandles:
        h.set_alpha(0.9)
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

    vec_batch_size = 25

    with open(RUN_FILE_APPROX, 'r') as f:
        runs_dict_approx = yaml.safe_load(f)

    with open(RUN_FILE_APPROX_NO_PRECOND, 'r') as f:
        runs_dict_approx_no_precond = yaml.safe_load(f)

    run_path_approx_mll = runs_dict_approx[angles][noise][vec_batch_size]['no_predcp']
    run_path_approx_map = runs_dict_approx[angles][noise][vec_batch_size]['predcp']
    run_path_approx_mll_no_precond = runs_dict_approx_no_precond[angles][noise][vec_batch_size]['no_predcp']
    run_path_approx_map_no_precond = runs_dict_approx_no_precond[angles][noise][vec_batch_size]['predcp']

    setting = 'angles_{}_noise_{}'.format(angles, int(noise*100))
    exp_name = 'calibration_uncertainty_estimates'

    num_samples = 5

    tb_paths_approx_mll = find_tensorboard_path_approx_mll(run_path_approx_mll, num_samples=num_samples, assert_vec_batch_size=vec_batch_size)
    tb_paths_approx_map = find_tensorboard_path_approx_map(run_path_approx_map, num_samples=num_samples, assert_vec_batch_size=vec_batch_size)
    tb_paths_approx_mll_no_precond = find_tensorboard_path_approx_mll(run_path_approx_mll_no_precond, num_samples=num_samples, assert_vec_batch_size=vec_batch_size)
    tb_paths_approx_map_no_precond = find_tensorboard_path_approx_map(run_path_approx_map_no_precond, num_samples=num_samples, assert_vec_batch_size=vec_batch_size)

    logs_approx_mll = [load_or_convert_log_file(p) for p in tb_paths_approx_mll]
    logs_approx_map = [load_or_convert_log_file(p) for p in tb_paths_approx_map]
    logs_approx_mll_no_precond = [load_or_convert_log_file(p) for p in tb_paths_approx_mll_no_precond]
    logs_approx_map_no_precond = [load_or_convert_log_file(p) for p in tb_paths_approx_map_no_precond]

    # exact and approx combined
    logs_list_mll = [logs_approx_mll_no_precond, logs_approx_mll]
    logs_list_map = [logs_approx_map_no_precond, logs_approx_map]

    method_label_list_mll = ['MLL w/o preconditioning', 'MLL w/ preconditioning']
    method_label_list_map = ['TV-MAP w/o preconditioning', 'TV-MAP w/ preconditioning']

    images_dir = OUT_PATH

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    color_mll_approx_list = ['#f552a3', '#9c55ff']
    color_map_approx_list = ['#e67330', '#6c6817']

    # kwargs_mll  = {
    #     'highlight_idx': highlight_idx,
    #     'kws_list': [
    #             {'color': color_mll_approx_list[0], 'alpha': 0.35, 'linewidth': 0.7},
    #             {'color': color_mll_approx_list[1], 'alpha': 0.35, 'linewidth': 0.7}],
    #     'add_highlight_kws_list': [
    #             {'alpha': .5, 'linewidth': 1.5},
    #             {'alpha': .5, 'linewidth': 1.5}],
    # }
    # kwargs_map = {
    #     'highlight_idx': highlight_idx,
    #     'kws_list': [
    #             {'color': color_map_approx_list[0], 'alpha': 0.35, 'linewidth': 0.7},
    #             {'color': color_map_approx_list[1], 'alpha': 0.35, 'linewidth': 0.7}],
    #     'add_highlight_kws_list': [
    #             {'alpha': .5, 'linewidth': 1.5},
    #             {'alpha': .5, 'linewidth': 1.5}],
    # }

    # fig = hyperparamters_fig_cg_residual(logs_list_mll, method_label_list_mll, **kwargs_mll)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_{}_mll.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_{}_mll.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)

    # fig = hyperparamters_fig_cg_residual(logs_list_map, method_label_list_map, **kwargs_map)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_{}_map.pdf'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0.)
    # fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_{}_map.png'.format(setting, highlight_idx)), bbox_inches='tight', pad_inches=0., dpi=600)


    kwargs_mll  = {
        'kws_list': [
                {'color': color_mll_approx_list[0], 'line_alpha': 0.75},
                {'color': color_mll_approx_list[1], 'line_alpha': 0.75}],
    }
    kwargs_map = {
        'kws_list': [
                {'color': color_map_approx_list[0], 'line_alpha': 0.75},
                {'color': color_map_approx_list[1], 'line_alpha': 0.75}],
    }

    fig = hyperparamters_fig_cg_residual(logs_list_mll, method_label_list_mll, **kwargs_mll)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_mll.pdf'.format(setting)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_mll.png'.format(setting)), bbox_inches='tight', pad_inches=0., dpi=600)

    fig = hyperparamters_fig_cg_residual(logs_list_map, method_label_list_map, **kwargs_map)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_map.pdf'.format(setting)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_vs_approx_cg_residual_{}_map.png'.format(setting)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()

if __name__ == "__main__": 
    plot()
