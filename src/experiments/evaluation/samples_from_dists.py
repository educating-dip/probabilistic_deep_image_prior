import sys
sys.path.append('../')

import os
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from itertools import islice
import pickle

image_dic= {
    'n_rows': 1, 
    'n_cols': 3, 
    'figsize': (10, 2.5),
    'hist': {
        'num_bins': [15, 10, 10],
        'num_k_retained': 5,  
        'zorder': [10, 8, 6, 4, 2, 0],
        'color': [['#2D4263', '#D9534F', '#3F0071', '#35DCDC', '#5A6C17', '#E2703A'], ['#2D4263', '#D9534F', '#3F0071', '#E2703A'],['#35DCDC', '#5A6C17', '#E2703A']], 
        'linewidth': 1.5, 
    }
}

def hex_to_rgb(value, alpha):
    value = value.lstrip('#')
    lv = len(value)
    out = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    out = [el / 255 for el in out] + [alpha]
    return tuple(out) 

def plot_hist_from_samples(data, legends, filename):

    fs_m1 = 10  # for figure ticks
    fs = 14  # for regular figure text
    fs_p1 = 24 #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)

    fig, axs = plt.subplots(image_dic['n_rows'], image_dic['n_cols'], figsize=image_dic['figsize'],
            facecolor='w', edgecolor='k', constrained_layout=True)

    kws = dict(histtype= "stepfilled", linewidth = image_dic['hist']['linewidth'], ls='dashed', density=True)
    for i, (ax, plot_data, plot_legends, plot_color) in enumerate(zip(axs, data, legends, image_dic['hist']['color'])): 
        for _, (el,  legend, zorder, color) in enumerate(zip(plot_data, plot_legends, image_dic['hist']['zorder'], plot_color)):
            ax.hist(el['tv'].flatten(), bins=image_dic['hist']['num_bins'][i], zorder=zorder, facecolor=hex_to_rgb(color, 0.45), edgecolor=hex_to_rgb(color, alpha=1), label=legend, **kws)
        if i == 0: ax.set_ylabel('density')
        ax.set_xlabel('average TV')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.3)
        ax.set_yscale('log')

        if i == 0: fig.legend(loc = 'lower center', ncol=3, bbox_to_anchor=(0.5, -0.25))
    fig.savefig(filename + '.png', dpi=100,  bbox_inches='tight')
    fig.savefig(filename + '.pdf',  bbox_inches='tight')


def plot_samples_from_dist(data, legends, scales=None, filename=''):

    fs_m1 = 8  # for figure ticks
    fs = 10  # for regular figure text
    fs_p1 = 12 #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title

    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)

    fig, axs = plt.subplots(1, 6, figsize=(11, 2),
            facecolor='w', edgecolor='k',  gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 1], 'wspace': 0.05},  constrained_layout=True)

    kws = dict(histtype= "stepfilled", linewidth = image_dic['hist']['linewidth'], ls='dashed', density=True)
    for i, (ax, plot_data, max_scale) in enumerate(zip(axs.flat, data, scales)):
        if i == 0:
            for _, (el,  legend, zorder, color) in enumerate(zip(plot_data, legends, [10, 8, 6, 4, 2, 0], ['#2D4263', '#D9534F', '#3F0071', '#35DCDC', '#5A6C17', '#E2703A'])):
                ax.hist(el['tv'].flatten(), bins=15, zorder=zorder, facecolor=hex_to_rgb(color, 0.45), edgecolor=hex_to_rgb(color, alpha=1), label=legend, **kws)
            if i == 0: ax.set_ylabel('density')
            ax.set_xlabel('average TV')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)
            ax.set_yscale('log')
            ax.set_aspect( (ax.get_xlim()[1]-ax.get_xlim()[0]) / ( np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])) )
            ax.legend(ncol=1, bbox_to_anchor= (-0.35, 0.85), frameon=False)

        else:
            if max_scale != 0: 
                ax.imshow(plot_data['x'][27].reshape(28,28), cmap='gray', vmin=0, vmax=max_scale)
            else: 
                ax.imshow(plot_data['x'][0].reshape(28,28), vmin=0, vmax=1, cmap='gray')
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.set_title(legends[i-1], y=1.01)

    fig.savefig(filename + '.png', dpi=100,  bbox_inches='tight')
    fig.savefig(filename + '.pdf',  bbox_inches='tight')

def n_tv_entries(side):
    return 2 * (side - 1) * side

def TV(x):
    h_tv = np.abs(np.diff(x, axis=-1, n=1)).sum()
    v_tv = np.abs(np.diff(x, axis=-2, n=1)).sum()
    return h_tv + v_tv

def draw_samples_from_Gaussian(loc, covariance_matrix):
    torch.manual_seed(2)
    covariance_matrix[np.diag_indices(covariance_matrix.shape[0])] += 1e-6
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=loc.flatten(), 
        scale_tril=torch.linalg.cholesky(covariance_matrix) 
        )
    samples = dist.sample((500, ))
    # return samples.clamp_(min=0, max=1).numpy()
    return samples.numpy()

def gather_data_from_bayes_dip(path_to_data, idx):
    
    data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)
    recon = data['recon'].squeeze().astype(np.float32)
    obs = data['observation'].squeeze().astype(np.float32)
    pred_cov_matrix_mll = data['model_post_cov_no_predcp'].astype(np.float32) 
    pred_cov_matrix_tv_map = data['model_post_cov_predcp'].astype(np.float32) 
    prior_cov_matrix_mll = data['Kxx_no_predcp'].astype(np.float32) 
    prior_cov_matrix_tv_map = data['Kxx_predcp'].astype(np.float32) 
    return (obs, recon, pred_cov_matrix_mll, pred_cov_matrix_tv_map, prior_cov_matrix_mll, prior_cov_matrix_tv_map)

def gather_priors_data_from_HMC_experiments():

    path_to_priors_data = 'good_HMC/prior_HMC/prior_samples.pickle'  # TODO insert absolute path if needed
    file = open(path_to_priors_data, 'rb')
    prior_sample_dict = pickle.load(file, encoding='unicode_escape')
    return {'samples_from_tv': prior_sample_dict['TV_samples'], 
                'samples_from_hybrid': prior_sample_dict['Hybrid_samples'], 
                'samples_from_gauss':  prior_sample_dict['Gaussian_samples']}


def gather_posterior_data_from_HMC_experiments(path_to_data, name_folder, method=''):

    path = os.path.join(path_to_data, name_folder)
    samples = {method: {'x': [], 'tv': []}}
    for idx in range(50):
        if idx == 17: # run failed
            continue
        file = open(os.path.join(path, 'test_result_dict_{}.pickle'.format(idx)), 'rb')
        dict = pickle.load(file)
        samples[method]['x'].append(dict[f'im_{idx}_samples']['x'])
        samples[method]['tv'].append(dict[f'im_{idx}_samples']['tv'])

    samples[method]['x'] =  np.concatenate(samples[method]['x'])
    samples[method]['tv'] =  np.concatenate(samples[method]['tv'])
    return samples

def gather_posterior_data_from_HMC_experiment_(path_to_data, name_folder, method=''):

    path = os.path.join(path_to_data, name_folder)
    samples = {method: {'x': [], 'tv': []}}
    file = open(os.path.join(path, 'test_result_dict.pickle'), 'rb')
    dict = pickle.load(file)
    for idx in range(50):

        samples[method]['x'].append(dict[f'im_{idx}_samples']['x'])
        samples[method]['tv'].append(dict[f'im_{idx}_samples']['tv'])

    samples[method]['x'] =  np.concatenate(samples[method]['x'])
    samples[method]['tv'] =  np.concatenate(samples[method]['tv'])
    return samples


def load_testset_KMNIST_dataset(path='kmnist', batchsize=1,
                               crop=False):
    # path = os.path.join('', path)  # TODO insert absolute path if needed
    testset = datasets.KMNIST(root=path, train=False, download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def gather_example_image_samples():
    samples = []
    kmnist_loader = load_testset_KMNIST_dataset()
    for _, data_sample in enumerate(islice(kmnist_loader, 10000)):
        data_sample = data_sample[0].squeeze()
        data_sample = (data_sample - data_sample.min()) / (data_sample.max() - data_sample.min())
        samples.append(data_sample.numpy())
    return np.stack(samples)

def apply_tv_to_deep_samples(samples):
    
    samples = samples.reshape(samples.shape[0], 28, 28)
    tv = []
    for sample in samples: 
        tv.append(TV(sample) / n_tv_entries(28))
    return {'tv': np.asarray(tv)}

if __name__ == "__main__":

    dic = {
        'image': 0, 
        'num_angles': 30, 
        'stddev': 0.05, 
    }

    DIRPATH='src/experiments/evaluation/'  # TODO insert absolute path if needed
    runs = OmegaConf.load(os.path.join(DIRPATH, 'kmnist_refined_tv_strength.yaml')) #  'runs.yaml'))
    path_to_data = runs[dic['num_angles']][dic['stddev']] # kmnist[num_angles][stddev][0]['path'] runs.kmnist[dic['num_angles']][dic['stddev']][0]['path']
    exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))

    name_folder = 'kmnist_num_angles_{}_stddev_{}'.format(dic['num_angles'], dic['stddev'])
    images_dir = os.path.join(DIRPATH, 'drawing_samples', name_folder)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    prior_samples = {'mll': [], 'map':[]}
    for idx in range(50):
        (obs, recon, pred_cov_matrix_mll, pred_cov_matrix_tv_map, prior_cov_matrix_mll, prior_cov_matrix_tv_map) = gather_data_from_bayes_dip(path_to_data, idx)

        breakpoint()
        if idx == 0:
            v_max_mll = 2 * np.mean(prior_cov_matrix_mll[np.diag_indices(784)] **0.5 )
            v_max_map = 2 * np.mean(prior_cov_matrix_tv_map[np.diag_indices(784)] **0.5 ) 

        prior_samples['mll'].append(draw_samples_from_Gaussian(torch.zeros(*recon.shape), torch.from_numpy(prior_cov_matrix_mll)))
        prior_samples['map'].append(draw_samples_from_Gaussian(torch.zeros(*recon.shape), torch.from_numpy(prior_cov_matrix_tv_map)))

    priors_hmc_exp_samples = gather_priors_data_from_HMC_experiments()
    image_samples = gather_example_image_samples()

    posterior_samples = {'mll': [], 'map':[]}
    path_to_data = runs[5][0.1]
    for idx in range(50):
        (obs, recon, pred_cov_matrix_mll, pred_cov_matrix_tv_map, prior_cov_matrix_mll, prior_cov_matrix_tv_map) = gather_data_from_bayes_dip(path_to_data, idx)

        posterior_samples['mll'].append(draw_samples_from_Gaussian(torch.from_numpy(recon), torch.from_numpy(pred_cov_matrix_mll)))
        posterior_samples['map'].append(draw_samples_from_Gaussian(torch.from_numpy(recon), torch.from_numpy(pred_cov_matrix_tv_map)))

    posterior_tv_samples = gather_posterior_data_from_HMC_experiment_(
        path_to_data='good_HMC/posterior_samples',  # TODO insert absolute path if needed
        name_folder='TV_HMC_{}_{}'.format(5, 0.1).replace('.', ''), 
        method='samples_from_tv')

    posterior_gaussian_samples = gather_posterior_data_from_HMC_experiment_(
        path_to_data='good_HMC/posterior_samples/',  # TODO insert absolute path if needed
        name_folder='Gaussian_HMC_{}_{}'.format(5, 0.1).replace('.', ''), 
        method='samples_from_gauss')

    posterior_hybrid_samples = gather_posterior_data_from_HMC_experiments(
        path_to_data='good_HMC/Hybrid_data/',  # TODO insert absolute path if needed
        name_folder='Hybrid_HMC_{}_{}'.format(5, 0.1).replace('.', ''), 
        method= 'samples_from_hybrid')

    plot_hist_from_samples(
        (
            (
                priors_hmc_exp_samples['samples_from_tv'], 
                priors_hmc_exp_samples['samples_from_hybrid'], 
                priors_hmc_exp_samples['samples_from_gauss'],
                apply_tv_to_deep_samples(np.clip(np.concatenate(prior_samples['mll']), a_min=0, a_max=1)), 
                apply_tv_to_deep_samples(np.clip(np.concatenate(prior_samples['map']), a_min=0, a_max=1)), 
                apply_tv_to_deep_samples(image_samples)
            ), 
            ( 
                posterior_tv_samples['samples_from_tv'], 
                posterior_hybrid_samples['samples_from_hybrid'], 
                posterior_gaussian_samples['samples_from_gauss'],
                apply_tv_to_deep_samples(image_samples)
            ),
            (
                apply_tv_to_deep_samples(np.clip(np.concatenate(posterior_samples['mll']), a_min=0, a_max=1)), 
                apply_tv_to_deep_samples(np.clip(np.concatenate(posterior_samples['map']), a_min=0, a_max=1)), 
                apply_tv_to_deep_samples(image_samples),
            )
        ), 
        (
            ('TV', 'TV-PredCP', 'Fact. Gauss.', 'Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'KMNIST'),
            ('TV', 'TV-PredCP', 'Fact. Gauss.', 'KMNIST'),
            ('Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'KMNIST'),
        ),
        filename= os.path.join(images_dir, 'samples_tv_hist_{}_{}'.format(dic['num_angles'], dic['stddev']).replace('.', ''))
    )

    plot_samples_from_dist(

        (

        (
            priors_hmc_exp_samples['samples_from_tv'], 
            priors_hmc_exp_samples['samples_from_hybrid'], 
            priors_hmc_exp_samples['samples_from_gauss'],
            apply_tv_to_deep_samples(np.clip(np.concatenate(prior_samples['mll']), a_min=0, a_max=1)), 
            apply_tv_to_deep_samples(np.clip(np.concatenate(prior_samples['map']), a_min=0, a_max=1)), 
            apply_tv_to_deep_samples(image_samples)
        ),
    
        priors_hmc_exp_samples['samples_from_tv'], 
        priors_hmc_exp_samples['samples_from_hybrid'], 
        priors_hmc_exp_samples['samples_from_gauss'],
        {'x': np.concatenate(prior_samples['mll'])}, 
        {'x': np.concatenate(prior_samples['map'])},

        ),
        ('TV', 'TV-PredCP', 'Fact. Gauss.', 'Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'KMNIST'),
        (0, 0, 0, 0, v_max_mll, v_max_map),
        filename= os.path.join(images_dir, 'prior_samples_{}_{}'.format(dic['num_angles'], dic['stddev']).replace('.', ''))

    )

    plot_samples_from_dist(

        (

        (
            posterior_tv_samples['samples_from_tv'], 
            posterior_hybrid_samples['samples_from_hybrid'], 
            posterior_gaussian_samples['samples_from_gauss'],
            apply_tv_to_deep_samples(np.clip(np.concatenate(posterior_samples['mll']), a_min=0, a_max=1)), 
            apply_tv_to_deep_samples(np.clip(np.concatenate(posterior_samples['map']), a_min=0, a_max=1)), 
            apply_tv_to_deep_samples(image_samples)
        ),
    
        posterior_tv_samples['samples_from_tv'], 
        posterior_hybrid_samples['samples_from_hybrid'], 
        posterior_gaussian_samples['samples_from_gauss'],
        {'x': np.concatenate(posterior_samples['mll'])}, 
        {'x': np.concatenate(posterior_samples['map'])},

        ),
        ('TV', 'TV-PredCP', 'Fact. Gauss.', 'Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)', 'KMNIST'),
        (0, 0, 0, 0, 0, 0),
        filename= os.path.join(images_dir, 'posterior_samples_{}_{}'.format(dic['num_angles'], dic['stddev']).replace('.', ''))

    )