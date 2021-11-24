import os
import numpy as np 
import bios
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

DIR_PATH='/media/chen/Res/dip_bayesian_ext/src/experiments' 
name = 'kmnist_sparse_10'
exp_name = 'calibration_uncertainty_estimates'
idx = 0 

def tensorboard_fig_subplots(data, filename):

    fs_m1 = 6  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 10  #  figure titles

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

    mll_list, map_list = data

    fig, axs = plt.subplots(1, 5, figsize=(7, 2),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = 1250
    mll_kws = dict(linestyle='solid', linewidth=0.85, alpha=0.75)
    map_kws = dict(linestyle='dashed', linewidth=0.85, alpha=0.75)
    name_list = ['negative_map_mll_scalar',  'predcp_scalars', 'posterior_hess_log_det_y_space_scalars', 'weight_prior_log_prob_scalars',  'reconstruction_log_lik_scalars']
    label_list = ['${\mathcal{G}}(\sigma^{2}_{y}, \\boldsymbol{\ell}, \\boldsymbol{\sigma}_{\\boldsymbol{\\theta}}^{2})$',
         '${\\rm log } p(\\boldsymbol{\ell})$', '${\\rm log} |H|$', '${\\rm log} p(\\boldsymbol{\\theta}^*)$', '${\\rm log} p(\mathbf{y}_{\delta}|\\boldsymbol{\\theta}^*)$']
    for i, ax in enumerate(axs.flatten()):
        mll_tmp = []
        map_tmp= []
        for mll_dic, map_dic in zip(mll_list, map_list):
            mll_tmp.append([v for k, v in mll_dic.items() if (name_list[i] in k and 'steps' not in k)])
            map_tmp.append([v for k, v in map_dic.items() if (name_list[i] in k and 'steps' not in k)])
        for k, (mll, map) in enumerate(zip(mll_tmp, map_tmp)):
            if k == 0: 
                ax.plot(list(range(N)), mll[0], color='red', zorder=10, **mll_kws, label='MLL')
                ax.plot(list(range(N)), map[0], color='red', zorder=10, **map_kws, label='Type-II MAP')
                ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                if i == 0: 
                    lgd = ax.legend(loc = 'upper right')
                    lgd.legendHandles[0].set_color('black')
                    lgd.legendHandles[1].set_color('black')
            ax.plot(list(range(N)), mll[0], color='gray', zorder=0, **mll_kws)
            ax.plot(list(range(N)), map[0], color='gray', zorder=0, **map_kws)
        ax.set_title(label_list[i], pad=15)
        ax.grid(0.25)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


def hyperparamters_fig_subplots(data, filename):

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

    mll_list, map_list = data

    fig, axs = plt.subplots(2, 4, figsize=(6, 3),
          facecolor='w', edgecolor='k', constrained_layout=True)
    N = 1250
    mll_kws = dict(linestyle='solid', linewidth=0.85, alpha=0.75)
    map_kws = dict(linestyle='dashed', linewidth=0.85, alpha=0.75)

    for i, ax in enumerate(axs.flatten()):
        mll_lengthscales = []
        map_lengthscales = []
        if i < 4: 
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_lengthscales.append([v for k, v in mll_dic.items() if ('lengthscale_{}'.format(i) in k and 'steps' not in k)])
                map_lengthscales.append([v for k, v in map_dic.items() if ('lengthscale_{}'.format(i) in k and 'steps' not in k)])
            for k, (mll_lengthscale, map_lengthscale) in enumerate(zip(mll_lengthscales, map_lengthscales)):
                if k == 0: 
                    ax.plot(list(range(N)), mll_lengthscale[0], color='red', zorder=10, **mll_kws, label='MLL')
                    ax.plot(list(range(N)), map_lengthscale[0], color='red', zorder=10, **map_kws, label='Type-II MAP')
                    if i == 0: 
                        lgd = ax.legend(loc = 'upper right')
                        lgd.legendHandles[0].set_color('black')
                        lgd.legendHandles[1].set_color('black')
                ax.plot(list(range(N)), mll_lengthscale[0], color='gray', zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_lengthscale[0], color='gray', zorder=0, **map_kws)
                ax.set_xticklabels([' ', ' ', ' ', ' '])  
            ax.set_title('$\ell_{}$'.format(i))
            ax.grid(0.25)
        mll_variances = []
        map_variances = []
        if i >= 4: 
            for mll_dic, map_dic in zip(mll_list, map_list):
                mll_variances.append([v for k, v in mll_dic.items() if ('variance_{}'.format(i-4) in k and 'steps' not in k)])
                map_variances.append([v for k, v in map_dic.items() if ('variance_{}'.format(i-4) in k and 'steps' not in k)])
            for k, (mll_variance, map_variance) in enumerate(zip(mll_variances, map_variances)):
                if k == 0: 
                    ax.plot(list(range(N)), mll_variance[0], color='red', zorder=10, **mll_kws)
                    ax.plot(list(range(N)), map_variance[0], color='red', zorder=10, **map_kws)
                ax.plot(list(range(N)), mll_variance[0], color='gray', zorder=0, **mll_kws)
                ax.plot(list(range(N)), map_variance[0], color='gray', zorder=0, **map_kws)   
            ax.set_title('$\\sigma^{%d}_{\\boldsymbol{\\theta}, %d}$' % (2, i - 4) ) 
            ax.grid(0.25)

    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


def extract_tensorboard_scalars(log_file=None, save_as_npz=None, tags=None):
    if not TF_AVAILABLE:
        raise RuntimeError('Tensorflow could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

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

def extract_tb_data(exp_tb_name):

    run_paths = bios.read(os.path.join(DIR_PATH, 'evaluation/runs.yaml'))
    path_to_folder = run_paths[exp_name][name][idx]['path']
    paths_to_tb = glob.glob(os.path.join(DIR_PATH, path_to_folder, exp_tb_name))
    extract_data_list = []
    for path in paths_to_tb: 
        extract_data_list.append(extract_tensorboard_scalars(path))
    
    return extract_data_list

mll_extract_data = extract_tb_data(
    exp_tb_name = "mrglik_opt_no_PredCP_recon_num_*/events*"
    )
map_extract_data = extract_tb_data(
    exp_tb_name = "mrglik_opt_w_PredCP_recon_num_*/events*"
    )

images_dir = os.path.join('./', 'tensorboard')

if not os.path.isdir(images_dir):
    os.makedirs(images_dir)

# hyperparamters_fig_subplots(
#     (mll_extract_data, map_extract_data),
#     images_dir + '/hyperparams_optim'
#     )

tensorboard_fig_subplots(
    (mll_extract_data, map_extract_data),
    images_dir + '/tensorboard_optim'
)