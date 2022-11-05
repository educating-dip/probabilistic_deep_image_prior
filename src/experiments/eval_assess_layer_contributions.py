import os
from math import ceil
from itertools import islice
import numpy as np
import hydra
from omegaconf import OmegaConf, DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import scipy
from tqdm import tqdm
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel, BlocksGPpriors
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, get_jac_fwAD_batch_ensemble,
        sample_from_posterior, approx_predictive_cov_image_block_from_samples_batched, predictive_image_block_log_prob_batched,
        get_image_block_mask_inds, stabilize_predictive_cov_image_block,
        stabilize_prior_cov_obs_mat, clamp_params, sample_from_posterior_via_jac, vec_weight_prior_cov_mul, 
        get_batched_jac_low_rank, get_reduced_model, get_inactive_and_leaf_modules_unet,
        get_prior_cov_obs_mat, get_prior_cov_obs_mat_jac_low_rank, fwAD_JvP_batch_ensemble, finite_diff_JvP_batch_ensemble,
        vec_jac_mul_batch
        )
from scalable_linearised_laplace.mc_pred_cp_loss import _sample_from_prior_over_weights
from dataset.walnut import get_inner_block_indices, INNER_PART_START_0, INNER_PART_START_1, INNER_PART_END_0, INNER_PART_END_1

def plot_var(var, var_gp_priors, var_normal_priors, gp_titles, normal_titles, prior_var_gp_priors=None, prior_var_normal_priors=None):
    import matplotlib
    import matplotlib.pyplot as plt

    fs_m1 = 6  # for figure ticks
    fs = 9  # for regular figure text
    fs_p1 = 12  #  figure titles

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

    fig, ax = plt.subplots(figsize=(8,5))

    var_numpy = var[:, :, INNER_PART_START_0:INNER_PART_END_0, INNER_PART_START_1:INNER_PART_END_1].flatten().detach().cpu().numpy()
    var_gp_priors_numpy = [var_prior[:, :, INNER_PART_START_0:INNER_PART_END_0, INNER_PART_START_1:INNER_PART_END_1].flatten().detach().cpu().numpy() for var_prior in var_gp_priors]
    var_normal_priors_numpy = [var_prior[:, :, INNER_PART_START_0:INNER_PART_END_0, INNER_PART_START_1:INNER_PART_END_1].flatten().detach().cpu().numpy() for var_prior in var_normal_priors]

    # import seaborn as sns

    # import pandas as pd
    # d = {'part': [], 'var': []}
    # d['part'] += ['full_net' for _ in var_numpy]
    # d['var'] += var_numpy.tolist()
    # for var_prior_numpy, title in zip(var_gp_priors_numpy, gp_titles):
    #     d['part'] += [str(title) for _ in var_prior_numpy]
    #     d['var'] += var_prior_numpy.tolist()
    # for var_prior_numpy, title in zip(var_normal_priors_numpy, normal_titles):
    #     d['part'] += [str(title) for _ in var_prior_numpy]
    #     d['var'] += var_prior_numpy.tolist()
    # data = pd.DataFrame(d)

    # sns.swarmplot(x='part', y='var', data=data, ax=ax)

    mean_var, std_var, min_var, max_var = np.mean(var_numpy), np.std(var_numpy), np.min(var_numpy), np.max(var_numpy)
    
    mean_var_gp_priors, std_var_gp_priors, min_var_gp_priors, max_var_gp_priors = (
        [np.mean(var_prior_numpy) for var_prior_numpy in var_gp_priors_numpy],
        [np.std(var_prior_numpy) for var_prior_numpy in var_gp_priors_numpy],
        [np.min(var_prior_numpy) for var_prior_numpy in var_gp_priors_numpy],
        [np.max(var_prior_numpy) for var_prior_numpy in var_gp_priors_numpy],
        )
    mean_var_normal_priors, std_var_normal_priors, min_var_normal_priors, max_var_normal_priors = (
        [np.mean(var_prior_numpy) for var_prior_numpy in var_normal_priors_numpy],
        [np.std(var_prior_numpy) for var_prior_numpy in var_normal_priors_numpy],
        [np.min(var_prior_numpy) for var_prior_numpy in var_normal_priors_numpy],
        [np.max(var_prior_numpy) for var_prior_numpy in var_normal_priors_numpy],
        )

    mean_var_list = [mean_var] + mean_var_gp_priors + mean_var_normal_priors
    std_var_list = [std_var] + std_var_gp_priors + std_var_normal_priors
    min_var_list = [min_var] + min_var_gp_priors + min_var_normal_priors
    max_var_list = [max_var] + max_var_gp_priors + max_var_normal_priors

    prior_var_list = [None] + (prior_var_gp_priors or [None] * len(mean_var_gp_priors)) + (prior_var_normal_priors or [None] * len(mean_var_normal_priors))

    labels = ['full net'] + gp_titles + normal_titles

    for i, (mean_var, std_var, min_var, max_var, prior_var) in enumerate(zip(mean_var_list, std_var_list, min_var_list, max_var_list, prior_var_list)):
        x = [i-0.25, i+0.25]
        ax.plot(x, [min_var] * 2, color='gray', linestyle='dashed', alpha=0.7)
        ax.plot(x, [max_var] * 2, color='gray', linestyle='dashed', alpha=0.7)
        ax.plot(x, [mean_var] * 2, color='blue', linewidth=2.5, alpha=1.)
        ax.fill_between(x, [mean_var - std_var]*2, [mean_var + std_var] * 2, color='blue', alpha=0.2)
        if prior_var is not None:
            ax.plot(x, [prior_var] * 2, color='crimson', linestyle='dotted', linewidth=1.5, alpha=1.)

    ax.set_xticks(range(len(mean_var_list)), labels=labels)
    ax.set_yscale('log')

    return fig, ax

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    elif cfg.name == 'walnut':
        loader = load_testset_walnut(cfg)
    else:
        raise NotImplementedError

    assert cfg.density.compute_single_predictive_cov_block.load_path is not None, "no previous run path specified (density.compute_single_predictive_cov_block.load_path)"
    # assert cfg.density.compute_single_predictive_cov_block.block_idx is not None, "no block index specified (density.compute_single_predictive_cov_block.block_idx)"

    load_path = cfg.density.compute_single_predictive_cov_block.load_path
    load_cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            example_image, _ = data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            if cfg.use_double:
                ray_trafos['ray_trafo_module'].to(torch.float64)
                ray_trafos['ray_trafo_module_adj'].to(torch.float64)
            observation, filtbackproj, example_image = simulate(
                example_image.double() if cfg.use_double else example_image, 
                ray_trafos,
                cfg.noise_specs
                )
            sample_dict = torch.load(os.path.join(load_path, 'sample_{}.pt'.format(i)), map_location=example_image.device)
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj, atol=1e-6)
            filtbackproj = sample_dict['filtbackproj']
            observation = sample_dict['observation']
            example_image = sample_dict['ground_truth']
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist']:
            # model from previous run
            dip_load_path = load_path if load_cfg.load_dip_models_from_path is None else load_cfg.load_dip_models_from_path
            path = os.path.join(dip_load_path, 'dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            # fine-tuned model
            path = os.path.join(get_original_cwd(), reconstructor.cfg.finetuned_params_path 
                if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
        else:
            raise NotImplementedError

        reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))

        with torch.no_grad():
            reconstructor.model.eval()
            recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
        recon = recon[0, 0].cpu().numpy()

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors,
            exclude_gp_priors_list=cfg.density.exclude_gp_priors_list, exclude_normal_priors_list=cfg.density.exclude_normal_priors_list)

        recon = torch.from_numpy(recon[None, None])
        if cfg.linearize_weights:
            linearized_weights_dict = torch.load(os.path.join(load_path, 'linearized_weights_{}.pt'.format(i)), map_location=reconstructor.device)
            linearized_weights = linearized_weights_dict['linearized_weights']
            lin_pred = linearized_weights_dict['linearized_prediction']

            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))

        else:
            linearized_weights = None
            lin_pred = None

        load_iter = cfg.density.compute_single_predictive_cov_block.get('load_mrglik_opt_iter', None)
        missing_keys, _ = bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i) if load_iter is None else 'bayesianized_model_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device), strict=False)
        assert not missing_keys
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i) if load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        mean_var_image_cov_priors = torch.load(cfg.density.eval_assess_layer_contributions.load_file_path, map_location='cpu')

        count = mean_var_image_cov_priors['count']
        var = mean_var_image_cov_priors['var']
        var_gp_priors = mean_var_image_cov_priors['var_gp_priors']
        var_normal_priors = mean_var_image_cov_priors['var_normal_priors']

        prior_var_gp_priors = [gp_prior.cov.log_variance.exp().item() for gp_prior in bayesianized_model.gp_priors]
        prior_var_normal_priors = [normal_prior.log_variance.exp().item() for normal_prior in bayesianized_model.normal_priors]

        print()
        print('prior variances')
        print('---------------')
        for j, prior_var_gp_prior in enumerate(prior_var_gp_priors):
            print('prior variance (GP {:d})'.format(j), prior_var_gp_prior)
        for j, prior_var_normal_prior in enumerate(prior_var_normal_priors):
            print('prior variance (Normal {:d})'.format(j), prior_var_normal_prior)

        print()
        print('estimated effective variances')
        print('-------------------')
        print('num_mc_samples:', count)
        print('mean variance (full jacobian):', torch.mean(var).item())
        print('max variance (full jacobian):', torch.max(var).item())
        for j, var_prior in enumerate(var_gp_priors):
            print('mean variance (GP {:d}):'.format(j), torch.mean(var_prior).item())
            print('max variance (GP {:d}):'.format(j), torch.max(var_prior).item())
        for j, var_prior in enumerate(var_normal_priors):
            print('mean variance (Normal {:d}):'.format(j), torch.mean(var_prior).item())
            print('max variance (Normal {:d}):'.format(j), torch.max(var_prior).item())

        print('sorted by mean estimated effective variance')
        print('-----------------------')
        print('GP:', sorted([('GP {:d}'.format(j), torch.mean(var_prior).item()) for j, var_prior in enumerate(var_gp_priors)], key=lambda x: x[1], reverse=True))
        print('Normal:', sorted([('Normal {:d}'.format(j), torch.mean(var_prior).item()) for j, var_prior in enumerate(var_normal_priors)], key=lambda x: x[1], reverse=True))
        print('GP + Normal:', sorted(
                [('GP {:d}'.format(j), torch.mean(var_prior).item()) for j, var_prior in enumerate(var_gp_priors)] +
                [('Normal {:d}'.format(j), torch.mean(var_prior).item()) for j, var_prior in enumerate(var_normal_priors)], key=lambda x: x[1], reverse=True))

        import matplotlib.pyplot as plt
        gp_sort_index = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        normal_sort_index=[0, 1, 2]
        var_gp_priors_sorted = [var_gp_priors[new_j] for new_j in gp_sort_index]
        var_normal_priors_sorted = [var_normal_priors[new_j] for new_j in normal_sort_index]
        prior_var_gp_priors_sorted = [prior_var_gp_priors[new_j] for new_j in gp_sort_index]
        prior_var_normal_priors_sorted = [prior_var_normal_priors[new_j] for new_j in normal_sort_index]
        fig, ax = plot_var(var, var_gp_priors_sorted, var_normal_priors_sorted,
                gp_titles=['\\texttt{In}'] + ['\\texttt{{Down{:d}}}'.format(i) for i in range(5)] + ['\\texttt{{Up{:d}}}'.format(i) for i in range(5)],
                normal_titles=['\\texttt{Skip0}', '\\texttt{Skip1}', '\\texttt{Out}'],
                prior_var_gp_priors=prior_var_gp_priors_sorted, prior_var_normal_priors=prior_var_normal_priors_sorted)
        ax.set_ylabel('Variance')
        ax.set_xlabel('Part of $J$')

        fig.savefig('layer_contributions_var.pdf', bbox_inches='tight', pad_inches=0.)
        plt.show()

if __name__ == '__main__':
    coordinator()
