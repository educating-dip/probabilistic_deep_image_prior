from itertools import islice
import hydra
import os
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch
import numpy as np
import scipy 
from hydra.utils import get_original_cwd
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut, load_trainset_MNIST_dataset, load_trainset_KMNIST_dataset)
from dataset.mnist import simulate
import tensorly as tl
tl.set_backend('pytorch')
from deep_image_prior import DeepImagePriorReconstructor, DeepImagePriorReconstructorSGLD
from deep_image_prior.utils import PSNR, SSIM, bayesianize_architecture, sample_from_bayesianized_model
from dataset import extract_trafos_as_matrices
from scalable_linearised_laplace import (approx_density_from_samples, approx_density_from_samples_mult_normal, approx_kernel_density, 
        approx_predictive_cov_image_block_from_samples, predictive_image_block_log_prob,
        get_image_block_masks, stabilize_predictive_cov_image_block, stabilize_prior_cov_obs_mat
        ) 

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
        if cfg.baseline.use_train_test:
            loader = load_trainset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
        if cfg.baseline.use_train_test:
            loader = load_trainset_KMNIST_dataset()
    elif cfg.name == 'walnut':
        loader = load_testset_walnut(cfg)
    else:
        raise NotImplementedError

    avg_image_metric, avg_test_log_lik, avg_test_log_lik_from_samples = [], [], []
    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue
        
        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            example_image, _ = data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            observation, filtbackproj, example_image = simulate(
                example_image,
                ray_trafos,
                cfg.noise_specs
                )
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)
        if cfg.baseline.name == 'sgld':
            reconstructor = DeepImagePriorReconstructorSGLD(**ray_trafo, cfg=cfg.net)

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        # pseudo-inverse computation
        if cfg.name in ['mnist', 'kmnist']:
            trafos = extract_trafos_as_matrices(ray_trafos)
            trafo = trafos[0].to(reconstructor.device)
            trafo_T_trafo = trafo.T @ trafo
            U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
            lik_hess_inv_diag_mean = (Vh.T @ torch.diag(1/S) @ U.T).diag().mean() # constructing noise in x correction term, sigma_y^2 = 1 (unit-var.)
  
        if cfg.name in ['mnist', 'kmnist']:
            num_samples = 10000
            if cfg.baseline.name == 'mcdo':
                bayesianize_architecture(reconstructor.model, p=cfg.baseline.p)
                DIRPATH='src/experiments/evaluation/kmnist_mcdo_baseline.yaml'  # TODO insert absolute path if needed
                path_to_model = os.path.join(OmegaConf.load(DIRPATH)[cfg.beam_num_angle][cfg.noise_specs.stddev], 'dip_model_{}.pt'.format(i))
                reconstructor.model.load_state_dict(torch.load(path_to_model, map_location=reconstructor.device))
                sample_recon = sample_from_bayesianized_model(reconstructor.model, filtbackproj.to(reconstructor.device), mc_samples=num_samples)
                mean = sample_recon.view(num_samples, -1).mean(dim=0)
                std = ( torch.var(sample_recon.view(num_samples, -1), dim=0) + lik_hess_inv_diag_mean) **.5
                log_prob_kernel_density = approx_kernel_density(example_image, sample_recon.cpu(), noise_x_correction_term=lik_hess_inv_diag_mean.cpu(), bw=cfg.baseline.bw) / example_image.numel()
                log_prob = approx_density_from_samples(mean, example_image.cuda(), sample_recon, noise_x_correction_term=lik_hess_inv_diag_mean) / example_image.numel()
                log_prob_from_samples_mult = approx_density_from_samples_mult_normal(mean, example_image.cuda(), sample_recon, noise_x_correction_term=lik_hess_inv_diag_mean) / example_image.numel()
            print('DIP reconstruction of sample {:d}'.format(i))
            print('PSNR:', PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('Test log-lik-kernel-density:', log_prob_kernel_density)
            print('Test log-lik-from-samples:', log_prob)
            print('Test log-lik-from-samples-cov:', log_prob_from_samples_mult)

            data = {
                    'filtbackproj': filtbackproj.cpu().numpy(), 
                    'image': example_image.cpu().numpy(), 
                    'recon': mean.cpu().numpy(),
                    'std': std.cpu().numpy(), 
                    'psnr': PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()), 
                    'test_log_likelihood': log_prob_kernel_density, 
                    'test_log_prob_from_samples': log_prob.cpu().numpy()
                    }

            np.savez('recon_info_{}'.format(i), **data)
            avg_image_metric.append(PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            avg_test_log_lik.append(log_prob_kernel_density)
            avg_test_log_lik_from_samples.append(log_prob.cpu().numpy())

        torch.save(reconstructor.model.state_dict(),
            './dip_model_{}.pt'.format(i))

    if cfg.name in ['kmnist', 'mnist']: 
        print('avg PSNR: ', np.mean(avg_image_metric))
        print('avg test log-lik: ', np.mean(avg_test_log_lik))
        print('avg test log-lik-from-samples: ', np.mean(avg_test_log_lik_from_samples))
        overall_data = {
                    'avg_image_metric': np.mean(avg_image_metric), 
                    'avg_test_log_likelihood': np.mean(avg_test_log_lik),
                    'avg_test_log_likelihood_from_samples': np.mean(avg_test_log_lik_from_samples)
                    }
        np.savez('overall_metrics{}'.format(i), **overall_data)


if __name__ == '__main__':
    coordinator()
