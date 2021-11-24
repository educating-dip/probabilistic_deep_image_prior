import os
import numpy as np
from numpy.core.defchararray import title 
import yaml
import torch
import bios
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
from create_recons_plots import collect_reconstruction_data
import matplotlib.gridspec as gridspec


DIR_PATH='/media/chen/Res/dip_bayesian_ext/src/experiments' 
run_paths = bios.read(os.path.join(DIR_PATH, 'evaluation/runs.yaml'))
name = 'kmnist_sparse_10' # kmnist_sparse_20
idx = 0 # [0, 1]
exp_name = 'samples'

def samples_from_dist_plot(samples, v_max=None, filename='', titles=''):

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

    samples_mll, samples_map = samples 
    v_max_mll, v_max_map = v_max if v_max is not None else (1, 1)

    fig, big_axs = plt.subplots(2, 1,
        figsize=(7.25, 3.5), gridspec_kw = {'wspace':0, 'hspace':0}, facecolor='w', edgecolor='k', constrained_layout=True)
    for big_ax, title in zip(big_axs, titles):
        big_ax.set_title(title)
        big_ax.axis('off')
    for i in range(5):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(samples_mll[i, :].reshape(28,28),
            vmin=0, vmax=v_max_mll, cmap='gray')
        ax.set_axis_off()
    shift = 5 + 1   
    for i in range(5):
        ax = fig.add_subplot(2,5,i + shift)
        ax.imshow(samples_map[i, :].reshape(28,28), 
            vmin=0, vmax=v_max_map,  cmap='gray')
        ax.set_axis_off()

    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


def samples_from_dist_joint_plot(samples_from_priors, samples_from_posterior, v_max=None, filename='', titles=''):

    fs_m1 = 6  # for figure ticks
    fs = 10    # for regular figure text
    fs_p1 = 15 #  figure titles

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

    samples_from_priors_mll, samples_from_priors_map = samples_from_priors 
    samples_from_posterior_mll, samples_from_posterior_map, = samples_from_posterior
    
    v_max_mll, v_max_map = v_max if v_max is not None else (1, 1)

    fig = plt.figure(figsize=(7.25, 3.5))
    # grid for pairs of subplots
    grid = plt.GridSpec(2, 2, wspace=0, hspace=0.1)

    # create fake subplot just to title pair of subplots
    fake = fig.add_subplot(grid[0])
    fake.set_title(' ', pad=20)
    fake.text(-0.1, 0.4, 'MLL', rotation=90)
    fake.set_axis_off()
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0], wspace=0)
    ax = fig.add_subplot(gs[0])
    ax.set_title('prior sample 1')
    ax.imshow(samples_from_priors_mll[2, :].reshape(28,28),
        vmin=0, vmax=v_max_mll, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[1])
    ax.set_title('posterior sample 1')
    ax.imshow(samples_from_posterior_mll[2, :].reshape(28,28),
        vmin=0, vmax=1, cmap='gray')    
    ax.set_axis_off()

    # create fake subplot just to title pair of subplots
    fake = fig.add_subplot(grid[1])
    fake.set_title(' ', pad=20)
    fake.set_axis_off()
    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[1], wspace=0)
    ax = fig.add_subplot(gs[0])
    ax.set_title('prior sample 2')
    ax.imshow(samples_from_priors_mll[4, :].reshape(28,28),
        vmin=0, vmax=v_max_mll, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[1])
    ax.set_title('posterior sample 2')
    ax.imshow(samples_from_posterior_mll[4, :].reshape(28,28),
        vmin=0, vmax=1, cmap='gray')    
    ax.set_axis_off()

    # create fake subplot just to title pair of subplots
    fake = fig.add_subplot(grid[2])
    fake.text(-0.1, 0.25, 'Type-II MAP', rotation=90)
    fake.set_axis_off()

    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[2], wspace=0)

    ax = fig.add_subplot(gs[0])
    ax.set_title(' ')
    ax.imshow(samples_from_priors_map[2, :].reshape(28,28),
        vmin=0, vmax=v_max_map, cmap='gray')
    ax.set_axis_off()
    ax = fig.add_subplot(gs[1])
    ax.imshow(samples_from_posterior_map[2, :].reshape(28,28),
        vmin=0, vmax=1, cmap='gray')
    ax.set_title(' ')
    ax.set_axis_off()

   # create fake subplot just to title pair of subplots
    fake = fig.add_subplot(grid[3])
    fake.set_axis_off()

    gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[3], wspace=0)
    ax = fig.add_subplot(gs[0])
    ax.imshow(samples_from_priors_map[4, :].reshape(28,28),
        vmin=0, vmax=v_max_map, cmap='gray')
    ax.set_title(' ')
    ax.set_axis_off()

    ax = fig.add_subplot(gs[1])
    ax.imshow(samples_from_posterior_map[4, :].reshape(28,28),
        vmin=0, vmax=1, cmap='gray')
    ax.set_title(' ')
    ax.set_axis_off()

    fig.savefig(filename + '.png', dpi=600)
    fig.savefig(filename + '.pdf')


if __name__ == "__main__": 

    seed = 2
    name_dir, path_to_data = run_paths[exp_name][name][idx]['name'], \
            run_paths[exp_name][name][idx]['path']
    (
    image, filtbackproj, recon, abs_error, 
    std_pred_mll, std_pred_map,
    covariance_matrix_mll, covariance_matrix_map, 
    Kxx_mll, Kxx_map 
    ) = \
    collect_reconstruction_data(
        path_to_data,
        0
        )

    images_dir = os.path.join('./', name_dir, 'samples')

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    mu_prior = torch.zeros(784)
    mu_post = torch.from_numpy(recon).flatten()
    torch.manual_seed(seed)
    cov_dist_mll = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=mu_post, 
        covariance_matrix=torch.from_numpy(covariance_matrix_mll)
        )
    model_pred_cov_samples_mll = cov_dist_mll.sample((100, )).numpy()
    torch.manual_seed(seed)
    cov_dist_map = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=mu_post,
        covariance_matrix=torch.from_numpy(covariance_matrix_map)
        )
    model_pred_cov_samples_map = cov_dist_map.sample((100, )).numpy()

    samples_from_dist_plot(
        (model_pred_cov_samples_mll, model_pred_cov_samples_map),
        filename=images_dir + '/samples_from_model_pred_cov',
        # titles=['posterior samples -- prior hyperparameters optimized with MLL', 'posterior samples -- prior hyperparameters optimized with type-II MAP']
        titles=[' ', ' ']
        )

    torch.manual_seed(seed)
    prior_dist_mll = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=mu_prior, 
        scale_tril=torch.linalg.cholesky(torch.from_numpy(Kxx_mll))
        )
    priors_samples_mll = prior_dist_mll.sample((100, )).numpy()
    torch.manual_seed(seed)
    prior_dist_map = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=mu_prior,
        scale_tril=torch.linalg.cholesky(torch.from_numpy(Kxx_map))
        )
    priors_samples_map = prior_dist_map.sample((100, )).numpy()
    v_max_mll = 2 * np.mean(Kxx_mll[np.diag_indices(784)]**0.5)
    v_max_map = 2 * np.mean(Kxx_map[np.diag_indices(784)]**0.5)

    samples_from_dist_plot(
        (priors_samples_mll, priors_samples_map),
        (v_max_mll, v_max_map),
        filename=images_dir + '/samples_from_priors', 
        #titles=['prior samples -- prior hyperparameters optimized with MLL', 'prior samples -- prior hyperparameters optimized with type-II MAP']
        titles=[' ', ' ']
        )

    samples_from_dist_joint_plot(
        (priors_samples_mll, priors_samples_map),
        (model_pred_cov_samples_mll, model_pred_cov_samples_map),
        (v_max_mll, v_max_map),
        filename=images_dir + '/samples_from_priors_and_posterior'
        )
