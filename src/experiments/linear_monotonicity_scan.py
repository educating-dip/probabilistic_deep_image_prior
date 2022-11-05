import hydra
import torch
import numpy as np
import random
import tensorly as tl
tl.set_backend('pytorch')
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch
from linearized_laplace import submatrix_image_space_lin_model_prior_cov
import matplotlib 
import matplotlib.pyplot as plt

def plot_monotonicity_traces(len_vec, overall_tv_samples_list, overall_tv_samples_list_norm, filename, label_list):

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
    # matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    fig, axs = plt.subplots(2, 5, figsize=(8, 4))
    for i, (ax, tv_samples_list, label) in enumerate(zip(axs.flatten(), overall_tv_samples_list, label_list)):

        ax.plot(len_vec, tv_samples_list, linewidth=2.5, color='#EC2215')
        ax.set_xscale('log')
        ax.grid(0.3)
        if i ==0: ax.set_ylabel('$\kappa$', fontsize=12)
        ax.set_title(label)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticklabels([' ', ' ', ' ', ' '])
        
        if i == 4: 
            break 
    
    for i, (ax, tv_samples_list) in enumerate(zip(axs.flatten()[i+1:], overall_tv_samples_list_norm)):

        ax.plot(len_vec, tv_samples_list, linewidth=2.5, color='#EC2215')
        ax.set_xscale('log')
        ax.grid(0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i == 0: ax.set_ylabel('$\kappa$', fontsize=12)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    fig.savefig(filename + '.pdf', bbox_inches='tight')
    fig.savefig(filename + '.png', dpi=600)

def compute_exptected_tv(reconstructor, recon, Jac_x, cfg, apply_normalization=False): 

    len_vec = np.logspace(-2, 2, 500)
    sweep_exp_tv_block_0, sweep_exp_tv_block_1, sweep_exp_tv_block_2, sweep_exp_tv_block_3, sweep_exp_tv_block_4 = [], [], [], [], []
    for lengthscale_init in tqdm(len_vec, leave=False):
        
        bayesianize_model = BayesianizeModel(
                reconstructor, **{
                    'lengthscale_init': lengthscale_init,
                    'variance_init': cfg.mrglik.priors.variance_init},
                    include_normal_priors=cfg.mrglik.priors.include_normal_priors)
        block_priors = BlocksGPpriors(
            reconstructor.model,
            bayesianize_model,
            reconstructor.device,
            lengthscale_init,
            cfg.mrglik.priors.variance_init,
            lin_weights=None)

        _, list_model_prior_cov = submatrix_image_space_lin_model_prior_cov(block_priors, Jac_x)
        expected_tv = []
        # reduce to gp priors only (omit normal priors)
        list_model_prior_cov = list_model_prior_cov[:len(bayesianize_model.gp_priors)]
        for cov in list_model_prior_cov:
            succed = False
            cnt = 0
            if apply_normalization: 
                A = cov.diag().sqrt().pow(-1).diag() # fixing marginal variance to be one
                cov = A @ cov @ A  
            while not succed:
                try: 
                    dist = \
                        torch.distributions.multivariate_normal.MultivariateNormal(loc=recon.flatten().to(block_priors.store_device),
                            scale_tril=torch.linalg.cholesky(cov))  # covariance_matrix=cov)
                    succed = True 
                except: 
                    cov[np.diag_indices(cov.shape[0])] += 1e-4
                    cnt += 1
                assert cnt < 100

            samples = dist.rsample((10000, ))
            expected_tv.append(tv_loss(samples.view(-1, *recon.shape)).detach().cpu().numpy() / 10000)
        sweep_exp_tv_block_0.append(expected_tv[0])
        sweep_exp_tv_block_1.append(expected_tv[1])
        sweep_exp_tv_block_2.append(expected_tv[2])
        sweep_exp_tv_block_3.append(expected_tv[3])
        sweep_exp_tv_block_4.append(expected_tv[4])

    return  sweep_exp_tv_block_0, sweep_exp_tv_block_1, sweep_exp_tv_block_2, sweep_exp_tv_block_3, sweep_exp_tv_block_4


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    np.random.seed(cfg.net.torch_manual_seed)
    random.seed(cfg.net.torch_manual_seed)

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError

    ray_trafos = get_standard_ray_trafos(cfg, return_op_mat=True)

    for i, (example_image, _) in enumerate(loader):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        # simulate and reconstruct the example image
        _, filtbackproj, example_image = simulate(example_image, 
            ray_trafos, cfg.noise_specs)
        dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 
                        'reco_space': ray_trafos['space']}
        reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)

        all_modules_under_prior = (
                BayesianizeModel(
                        reconstructor, **{
                            'lengthscale_init': cfg.mrglik.priors.lengthscale_init,
                            'variance_init': cfg.mrglik.priors.variance_init},
                            include_normal_priors=cfg.mrglik.priors.include_normal_priors)
                .get_all_modules_under_prior())
        
        # reconstruction - learning MAP estimate weights
        filtbackproj = filtbackproj.to(reconstructor.device)
        path_to_params = os.path.join('/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/scripts/outputs/2022-01-21T13:19:34.983951Z/dip_model_0.pt')
        reconstructor.model.load_state_dict(torch.load(path_to_params))
        recon = reconstructor.model.forward(filtbackproj)[0]
        
        # estimate the Jacobian
        Jac_x = compute_jacobian_single_batch(
            filtbackproj,
            reconstructor.model, 
            all_modules_under_prior,
            example_image.flatten().shape[0]
            )
        
        (sweep_exp_tv_block_0, sweep_exp_tv_block_1, sweep_exp_tv_block_2, sweep_exp_tv_block_3, sweep_exp_tv_block_4) = \
             compute_exptected_tv(reconstructor, recon, Jac_x, cfg, False)

        (sweep_exp_tv_block_0_norm, sweep_exp_tv_block_1_norm, sweep_exp_tv_block_2_norm, sweep_exp_tv_block_3_norm, sweep_exp_tv_block_4_norm) = \
             compute_exptected_tv(reconstructor, recon, Jac_x, cfg, True)

        plot_monotonicity_traces(
            np.logspace(-2, 2, 500), 
            (sweep_exp_tv_block_0, 
            sweep_exp_tv_block_1, 
            sweep_exp_tv_block_2, 
            sweep_exp_tv_block_3,
            sweep_exp_tv_block_4),
            (sweep_exp_tv_block_0_norm, 
            sweep_exp_tv_block_1_norm,
            sweep_exp_tv_block_2_norm, 
            sweep_exp_tv_block_3_norm, 
            sweep_exp_tv_block_3_norm),
            'linear_monotonicity_scan',
            ['$\ell_{0}$---\\texttt{In}', '$\ell_{1}$---\\texttt{Down}', '$\ell_{2}$--- \\texttt{Down}', '$\ell_{3}$ --- \\texttt{Up}', '$\ell_{4}$ --- \\texttt{Up}']
            )
        
        if i == 0: 
            break
                
if __name__ == '__main__':
    coordinator()