import os
import numpy as np
import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RUN_FILE_APPROX_EXACT_DENSITY = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx_exact_density_block_sizes.yaml'
RUN_FILE_APPROX_SAMPLES = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx_samples_eval.yaml'
DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext'
IMAGES_DIR='./images_kmnist'

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

def get_approx_log_prob_list(run_path, num_samples, ddof=0):
    log_probs = []
    for i in range(num_samples):
        predictive_image_log_prob_dict = torch.load(os.path.join(run_path, 'predictive_image_log_prob_{}.pt'.format(i)), map_location='cpu')
        num_pixels = np.sum([np.prod(mask_inds.shape) for mask_inds in predictive_image_log_prob_dict['block_mask_inds']])
        assert num_pixels == 784
        log_prob = torch.sum(torch.stack(predictive_image_log_prob_dict['block_log_probs'])).item() / num_pixels
        log_probs.append(log_prob)
    return log_probs

# def get_approx_log_prob_list_blocks(run_path, num_samples, block_size, ddof=0):
#     log_probs = []
#     for i in range(num_samples):
#         block_log_probs = []
#         block_mask_inds = []
#         for block_idx in range(784 // block_size**2):
#             predictive_image_log_prob_block_dict = torch.load(os.path.join(run_path, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i)), map_location='cpu')
#             block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'])
#             block_mask_inds.append(predictive_image_log_prob_block_dict['mask_inds'])
#         num_pixels = np.sum([np.prod(mask_inds.shape) for mask_inds in block_mask_inds])
#         assert num_pixels == 784
#         log_prob = torch.sum(torch.stack(block_log_probs)).item() / num_pixels
#         log_probs.append(log_prob)
#     return log_probs

if __name__ == '__main__':

    num_samples = 5

    log_probs_approx_exact_density = {}
    log_probs_approx_samples = {}

    noise_list = [0.05]

    method = 'tv_map'

    highlight_idx = 3

    vec_batch_size = 25

    block_size_list = [28, 14, 7, 4, 2, 1]
    num_mc_samples_list = [int(2**i) for i in range(5, 15)]

    with open(RUN_FILE_APPROX_EXACT_DENSITY, 'r') as f:
        runs_dict_approx_exact_density = yaml.safe_load(f)

    with open(RUN_FILE_APPROX_SAMPLES, 'r') as f:
        runs_dict_approx_samples = yaml.safe_load(f)

    angles = 20

    load_from_previous_run_path = None
    # load_from_previous_run_path = '/tmp/plot_data_num_mc_samples'
    if load_from_previous_run_path:
        plot_data = np.load(os.path.join(load_from_previous_run_path, 'plot_data.npz'), allow_pickle=True)
        log_probs_approx_exact_density = plot_data['log_probs_approx_exact_density'].item()
        log_probs_approx_samples = plot_data['log_probs_approx_samples'].item()
    else:
        for noise in noise_list:
            log_probs_approx_exact_density[(angles, noise)] = {}
            for i, block_size in enumerate(block_size_list):
                log_probs_approx_exact_density[(angles, noise)][block_size] = {}
                # for method in ['mll', 'tv_map']:
                run_path = runs_dict_approx_exact_density[angles][noise][vec_batch_size]['no_predcp' if method == 'mll' else 'predcp']
                run_path = os.path.join(DIR_PATH, 'multirun', run_path.split('/multirun/')[-1], str(i))  # translate to local path, choose respective sub-run
                cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
                job_name = OmegaConf.load(os.path.join(run_path, '.hydra', 'hydra.yaml')).hydra.job.name
                assert job_name == 'exact_density_for_bayes_dip'
                assert cfg.density.block_size_for_approx == block_size
                load_path = cfg.density.compute_single_predictive_cov_block.load_path
                cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))
                assert cfg.beam_num_angle == angles
                assert cfg.noise_specs.stddev == noise
                assert cfg.mrglik.optim.include_predcp == (method == 'tv_map')
                assert cfg.mrglik.impl.vec_batch_size == vec_batch_size
                log_probs_approx_exact_density[(angles, noise)][block_size][method] = get_approx_log_prob_list(run_path, num_samples=num_samples)

            log_probs_approx_samples[(angles, noise)] = {}
            for i, block_size in enumerate(block_size_list):
                log_probs_approx_samples[(angles, noise)][block_size] = {}
                for j, num_mc_samples in enumerate(num_mc_samples_list):
                    log_probs_approx_samples[(angles, noise)][block_size][num_mc_samples] = {}
                    # for method in ['mll', 'tv_map']:
                    run_path = runs_dict_approx_samples[angles][noise][vec_batch_size]['no_predcp' if method == 'mll' else 'predcp'][block_size]
                    run_path = os.path.join(DIR_PATH, 'multirun', run_path.split('/multirun/')[-1], str(j))  # translate to local path, choose respective sub-run
                    cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
                    job_name = OmegaConf.load(os.path.join(run_path, '.hydra', 'hydra.yaml')).hydra.job.name
                    assert job_name == 'estimate_density_from_samples'
                    # assert job_name == 'exact_density_for_bayes_dip'
                    assert cfg.density.block_size_for_approx == block_size
                    assert cfg.density.estimate_density_from_samples.limit_loaded_samples == num_mc_samples
                    load_path = cfg.density.compute_single_predictive_cov_block.load_path
                    cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))
                    assert cfg.beam_num_angle == angles
                    assert cfg.noise_specs.stddev == noise
                    assert cfg.mrglik.optim.include_predcp == (method == 'tv_map')
                    assert cfg.mrglik.impl.vec_batch_size == vec_batch_size
                    log_probs_approx_samples[(angles, noise)][block_size][num_mc_samples][method] = get_approx_log_prob_list(run_path, num_samples=num_samples)

        try:
            np.savez('/tmp/plot_data_num_mc_samples/plot_data.npz',
                **{'log_probs_approx_exact_density': log_probs_approx_exact_density, 'log_probs_approx_samples': log_probs_approx_samples})
        except:
            pass

    for noise in noise_list:
        print((angles, noise))
        print('log_probs_approx_exact_density', log_probs_approx_exact_density[(angles, noise)])
        print('log_probs_approx_samples', log_probs_approx_samples[(angles, noise)])


    fs_m1 = 6  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 9  #  figure titles

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

    images_dir = IMAGES_DIR
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)


    fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))
    axs = np.array([axs])
    for i, (noise, ax) in enumerate(zip(noise_list, axs.flat)):
        log_probs_approx_exact_density_per_block_size_and_sample_map = np.array(
                [log_probs_approx_exact_density[(angles, noise)][block_size][method] for block_size in block_size_list])  # block size x image
        log_probs_approx_samples_per_block_size_and_num_mc_samples_and_sample_map = np.array(
                [[log_probs_approx_samples[(angles, noise)][block_size][num_mc_samples][method] for num_mc_samples in num_mc_samples_list] for block_size in block_size_list])  # block size x num mc samples x image
        # ax.set_title('{}\,\% noise'.format(int(noise*100)))
        ax.set_xlabel('\# MC samples')
        ax.set_xscale('log', base=2)
        ax.xaxis.set_ticks(num_mc_samples_list)
        ax.grid(0.25)
        mean_log_probs_approx_exact_density_per_block_size_map = np.mean(log_probs_approx_exact_density_per_block_size_and_sample_map, axis=-1)
        std_log_probs_approx_exact_density_per_block_size_map = np.std(log_probs_approx_exact_density_per_block_size_and_sample_map, axis=-1) / np.sqrt(num_samples)
        mean_log_probs_approx_samples_per_block_size_and_num_mc_samples_map = np.mean(log_probs_approx_samples_per_block_size_and_num_mc_samples_and_sample_map, axis=-1)
        std_log_probs_approx_samples_per_block_size_and_num_mc_samples_map = np.std(log_probs_approx_samples_per_block_size_and_num_mc_samples_and_sample_map, axis=-1) / np.sqrt(num_samples)
        color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
        handles_exact = []
        handles_samples = []
        for j, (block_size, mean_log_probs_approx_exact_density_map, std_log_probs_approx_exact_density_map, mean_log_probs_approx_samples_per_num_mc_samples_map, std_log_probs_approx_samples_per_num_mc_samples_map) in enumerate(zip(
                    block_size_list, mean_log_probs_approx_exact_density_per_block_size_map, std_log_probs_approx_exact_density_per_block_size_map, mean_log_probs_approx_samples_per_block_size_and_num_mc_samples_map, std_log_probs_approx_samples_per_block_size_and_num_mc_samples_map)):
            h_exact_map, = errorfill(np.asarray([num_mc_samples_list[0], num_mc_samples_list[-1]]), np.asarray([mean_log_probs_approx_exact_density_map] * 2), np.asarray([std_log_probs_approx_exact_density_map] * 2),
                            color=color_list[j], alpha_fill=0.05, line_alpha=0.75, ax=ax, lw=1, linestyle='-', fill_linewidths=0.2)
            h_samples_map, = errorfill(num_mc_samples_list, mean_log_probs_approx_samples_per_num_mc_samples_map, std_log_probs_approx_samples_per_num_mc_samples_map,
                            color=color_list[j], alpha_fill=0.05, line_alpha=0.75, ax=ax, lw=1, linestyle='dashed', fill_linewidths=0.2)
            handles_exact.append(h_exact_map)
            handles_samples.append(h_samples_map)
        lgd = ax.legend(handles_samples[::-1], ['patch size ${}\\times {}$'.format(block_size, block_size) for block_size in block_size_list][::-1])
        ax.add_artist(lgd)
        lgd_kind = ax.legend([handles_exact[0], handles_samples[0]], ['exact', 'sampling'])
        ax.set_title('Bayes DIP ({})'.format({'tv_map': 'TV-MAP', 'mll': 'MLL'}[method]))
        for h in lgd_kind.legendHandles:
            h.set_color('black')
    ylim = [min([ax.get_ylim()[0] for ax in axs.flat]), max([ax.get_ylim()[1] for ax in axs.flat])]
    for ax in axs.flat:
        ax.set_ylim(ylim)
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[0].set_ylabel('test log-likelihood')

    fig.savefig(os.path.join(images_dir, 'kmnist_approx_num_mc_samples_error_bars_angles_{}_{}.pdf'.format(angles, {'mll': 'mll', 'tv_map': 'map'}[method])), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_approx_num_mc_samples_error_bars_angles_{}_{}.png'.format(angles, {'mll': 'mll', 'tv_map': 'map'}[method])), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()
