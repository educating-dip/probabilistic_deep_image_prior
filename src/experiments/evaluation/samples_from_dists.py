import sys
sys.path.append('../')

import os
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

image_dic= {
    'n_rows': 1, 
    'n_cols': 1, 
    'figsize': (4, 4),
}

def TV(x):
    h_tv = np.abs(np.diff(x, axis=-1, n=1)).sum()
    v_tv = np.abs(np.diff(x, axis=-2, n=1)).sum()
    return h_tv + v_tv

def plot_hist_from_samples(data, legends, filename):

    fig, axs = plt.subplots(image_dic['n_rows'], image_dic['n_cols'], figsize=image_dic['figsize'],
            facecolor='w', edgecolor='k', constrained_layout=True)

    for _, (el, legend) in enumerate(zip(data, legends)):
            axs.hist(el['tv'], density=True, label=legend, alpha=0.5)
    
    plt.legend()
    
    fig.savefig(filename + '.png', dpi=100)
    fig.savefig(filename + '.pdf')

def n_tv_entries(side):
    return 2 * (side - 1) * side

def draw_samples_from_Gaussian(loc, covariance_matrix):
    torch.manual_seed(2)
    covariance_matrix[np.diag_indices(covariance_matrix.shape[0])] += 1e6
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=loc.flatten(), 
        scale_tril=torch.linalg.cholesky(covariance_matrix) 
        )
    return dist.sample((500, )).numpy()


def gather_data_from_bayes_dip(path_to_data, idx):
    
    data = np.load(os.path.join(path_to_data, 'recon_info_{}.npz'.format(idx)),  allow_pickle=True)
    recon = data['recon'].squeeze().astype(np.float32) 
    pred_cov_matrix_mll = data['model_post_cov_no_predcp'].astype(np.float32) 
    pred_cov_matrix_tv_map = data['model_post_cov_predcp'].astype(np.float32) 
    prior_cov_matrix_mll = data['Kxx_no_predcp'].astype(np.float32) 
    prior_cov_matrix_tv_map = data['Kxx_predcp'].astype(np.float32) 
    return (recon, pred_cov_matrix_mll, pred_cov_matrix_tv_map, prior_cov_matrix_mll, prior_cov_matrix_tv_map)

def gather_priors_data_from_HMC_experiments():
    import pickle
    path_to_priors_data = '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/good_HMC/prior_HMC/prior_samples.pickle'
    file = open(path_to_priors_data, 'rb')
    prior_sample_dict = pickle.load(file, encoding='unicode_escape')
    return {'samples_from_tv': prior_sample_dict['TV_samples'], 
                'samples_from_hybrid': prior_sample_dict['Hybrid_samples'], 
                'samples_from_gauss':  prior_sample_dict['Gaussian_samples']}

def apply_tv_to_deep_samples(samples):
    samples = samples.reshape(500, 28, 28)
    tv = []
    for sample in samples: 
        tv.append(TV(sample) / n_tv_entries(28))
    return {'tv': np.asarray(tv)}

if __name__ == "__main__":

    dic = {
        'image': 0, 
        'num_angles': 20, 
        'stddev':0.05
    }

    DIRPATH='/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/dip_bayesian_ext/src/experiments/evaluation/'
    runs = OmegaConf.load(os.path.join(DIRPATH, 'runs.yaml'))
    path_to_data = runs.kmnist[dic['num_angles']][dic['stddev']][0]['path']
    exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
    (recon, pred_cov_matrix_mll, pred_cov_matrix_tv_map, prior_cov_matrix_mll, prior_cov_matrix_tv_map) = gather_data_from_bayes_dip(path_to_data, dic['image'])    
    prior_samples = {'mll': draw_samples_from_Gaussian(torch.zeros(*recon.shape), torch.from_numpy(prior_cov_matrix_mll)),
                        'map': draw_samples_from_Gaussian(torch.zeros(*recon.shape), torch.from_numpy(prior_cov_matrix_tv_map))
                    }
    posterior_samples = {'mll': draw_samples_from_Gaussian(torch.from_numpy(recon), torch.from_numpy(pred_cov_matrix_mll)),
                            'map': draw_samples_from_Gaussian(torch.from_numpy(recon), torch.from_numpy(pred_cov_matrix_tv_map))
                    }
    priors_hmc_exp_samples = gather_priors_data_from_HMC_experiments()
    name_folder = 'kmnist_num_angles_{}_stddev_{}'.format(dic['num_angles'], dic['stddev'])
    images_dir = os.path.join(DIRPATH, 'drawing_samples', name_folder)

    if not os.path.isdir(images_dir):
        os.makedirs(images_dir)

    plot_hist_from_samples(
        (priors_hmc_exp_samples['samples_from_tv'], 
                priors_hmc_exp_samples['samples_from_hybrid'], 
                priors_hmc_exp_samples['samples_from_gauss'],
                apply_tv_to_deep_samples(prior_samples['mll']), 
                apply_tv_to_deep_samples(prior_samples['map'])
        ), 
        ('TV', 'Hybrid', 'Gaussian', 'Bayes DIP (MLL)', 'Bayes DIP (TV-MAP)'),
        filename= os.path.join(images_dir, 'priors_samples_tv_hist')
    )
