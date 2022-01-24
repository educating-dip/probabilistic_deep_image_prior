import os
import numpy as np
import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RUN_FILE_APPROX = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx_exact_density_block_sizes.yaml'
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

def get_approx_log_prob_list(run_path, num_samples, ddof=1):  # using less biased estimate of std because we only take 5 samples (by default)
    log_probs = []
    for i in range(num_samples):
        predictive_image_log_prob_dict = torch.load(os.path.join(run_path, 'predictive_image_log_prob_{}.pt'.format(i)), map_location='cpu')
        num_pixels = np.sum([np.prod(mask_inds.shape) for mask_inds in predictive_image_log_prob_dict['block_mask_inds']])
        assert num_pixels == 784
        log_prob = torch.sum(torch.stack(predictive_image_log_prob_dict['block_log_probs'])).item() / num_pixels
        log_probs.append(log_prob)
    return log_probs

if __name__ == '__main__':

    num_samples = 5

    log_probs_approx = {}

    noise_list = [0.05, 0.1]

    highlight_idx = 3

    vec_batch_size = 25

    block_size_list = [28, 14, 7, 4, 2, 1]

    with open(RUN_FILE_APPROX, 'r') as f:
        runs_dict_approx = yaml.safe_load(f)

    angles = 20
    for noise in noise_list:
        log_probs_approx[(angles, noise)] = {}

        for i, block_size in enumerate(block_size_list):
            log_probs_approx[(angles, noise)][block_size] = {}
            for method in ['mll', 'tv_map']:
                run_path = runs_dict_approx[angles][noise][vec_batch_size]['no_predcp' if method == 'mll' else 'predcp']
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
                log_probs_approx[(angles, noise)][block_size][method] = get_approx_log_prob_list(run_path, num_samples=num_samples)

        print((angles, noise))
        print('approx', log_probs_approx[(angles, noise)])

    table = ''
    for noise in noise_list:
        table += ' & ' + '\multicolumn{{2}}{{c}}{{\\textbf{{{}\,\% noise}}}}'.format(int(noise * 100))
    table += '\\\\\n'
    table += 'block size'
    for noise in noise_list:
        table += ' & MLL & TV-MAP'
    table += '\\\\\n'

    for block_size in block_size_list:
        table += '${}\\times {}$'.format(block_size, block_size)
        for noise in noise_list:
            log_probs_approx_mll = log_probs_approx[(angles, noise)][block_size]['mll']
            log_probs_approx_map = log_probs_approx[(angles, noise)][block_size]['tv_map']
            mean_log_prob_approx_mll = np.mean(log_probs_approx_mll)
            std_error_log_prob_approx_mll = np.std(log_probs_approx_mll, ddof=1) / np.sqrt(num_samples)
            mean_log_prob_approx_map = np.mean(log_probs_approx_map)
            std_error_log_prob_approx_map = np.std(log_probs_approx_map, ddof=1) / np.sqrt(num_samples)
            table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_prob_approx_mll, std_error_log_prob_approx_mll)
            table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_prob_approx_map, std_error_log_prob_approx_map)
        table += '\\\\\n'

    print(table)


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

    images_dir = IMAGES_DIR
    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.25))

    for i, (noise, ax) in enumerate(zip(noise_list, axs.flat)):
        log_probs_approx_per_block_size_and_sample_mll = np.array([log_probs_approx[(angles, noise)][block_size]['mll'] for block_size in block_size_list])  # block size x samples
        log_probs_approx_per_block_size_and_sample_map = np.array([log_probs_approx[(angles, noise)][block_size]['tv_map'] for block_size in block_size_list])  # block size x samples
        ax.set_title('{}\,\% noise'.format(int(noise*100)))
        ax.set_xlabel('block size [px]')
        ax.set_xscale('log', base=2)
        ax.xaxis.set_ticks(block_size_list)
        ax.xaxis.set_major_formatter('${x}^2$')
        ax.grid(0.25)
        for j, (log_probs_approx_per_block_size_mll, log_probs_approx_per_block_size_map) in enumerate(zip(log_probs_approx_per_block_size_and_sample_mll.T, log_probs_approx_per_block_size_and_sample_map.T)):  # samples
            ax.plot(block_size_list, log_probs_approx_per_block_size_mll, color=color_mll, alpha=0.9, label='MLL' if j == 0 else None)
            ax.plot(block_size_list, log_probs_approx_per_block_size_map, color=color_map, alpha=0.9, label='TV-MAP' if j == 0 else None)
    ylim = [min([ax.get_ylim()[0] for ax in axs.flat]), max([ax.get_ylim()[1] for ax in axs.flat])]
    for ax in axs.flat:
        ax.set_ylim(ylim)
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[1].set_yticklabels([' '])
    axs[0].set_ylabel('log-likelihood')
    axs[1].legend()

    fig.savefig(os.path.join(images_dir, 'kmnist_approx_block_sizes_angles_{}.pdf'.format(angles)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_approx_block_sizes_angles_{}.png'.format(angles)), bbox_inches='tight', pad_inches=0., dpi=600)

    # plt.show()


    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.25))

    for i, (noise, ax) in enumerate(zip(noise_list, axs.flat)):
        log_probs_approx_per_block_size_and_sample_mll = np.array([log_probs_approx[(angles, noise)][block_size]['mll'] for block_size in block_size_list])  # block size x samples
        log_probs_approx_per_block_size_and_sample_map = np.array([log_probs_approx[(angles, noise)][block_size]['tv_map'] for block_size in block_size_list])  # block size x samples
        ax.set_title('{}\,\% noise'.format(int(noise*100)))
        ax.set_xlabel('block size [px]')
        ax.set_xscale('log', base=2)
        ax.xaxis.set_ticks(block_size_list)
        ax.xaxis.set_major_formatter('${x}^2$')
        ax.grid(0.25)
        mean_log_probs_approx_per_block_size_and_sample_mll = np.mean(log_probs_approx_per_block_size_and_sample_mll, axis=1)
        std_log_probs_approx_per_block_size_and_sample_mll = np.std(log_probs_approx_per_block_size_and_sample_mll, axis=1) / np.sqrt(num_samples)
        mean_log_probs_approx_per_block_size_and_sample_map = np.mean(log_probs_approx_per_block_size_and_sample_map, axis=1)
        std_log_probs_approx_per_block_size_and_sample_map = np.std(log_probs_approx_per_block_size_and_sample_map, axis=1) / np.sqrt(num_samples)
        h_mll, = errorfill(block_size_list, mean_log_probs_approx_per_block_size_and_sample_mll, std_log_probs_approx_per_block_size_and_sample_mll,
                        color=color_mll, alpha_fill=0.3, line_alpha=1, ax=ax, lw=1, linestyle='-', fill_linewidths=0.2)
        h_map, = errorfill(block_size_list, mean_log_probs_approx_per_block_size_and_sample_map, std_log_probs_approx_per_block_size_and_sample_map,
                        color=color_map, alpha_fill=0.3, line_alpha=1, ax=ax, lw=1, linestyle='-', fill_linewidths=0.2)
    ylim = [min([ax.get_ylim()[0] for ax in axs.flat]), max([ax.get_ylim()[1] for ax in axs.flat])]
    for ax in axs.flat:
        ax.set_ylim(ylim)
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axs[1].set_yticklabels([' '])
    axs[0].set_ylabel('log-likelihood')
    axs[1].legend([h_mll, h_map], ['MLL', 'TV-MAP'])

    fig.savefig(os.path.join(images_dir, 'kmnist_approx_block_sizes_error_bars_angles_{}.pdf'.format(angles)), bbox_inches='tight', pad_inches=0.)
    fig.savefig(os.path.join(images_dir, 'kmnist_approx_block_sizes_error_bars_angles_{}.png'.format(angles)), bbox_inches='tight', pad_inches=0., dpi=600)

    plt.show()
