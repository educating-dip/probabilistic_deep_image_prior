from itertools import islice
import hydra
import os
from omegaconf import DictConfig
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
from scalable_linearised_laplace import (approx_density_from_samples, approx_kernel_density, 
        approx_predictive_cov_image_block_from_samples, predictive_image_block_log_prob,
        get_image_block_masks, stabilize_predictive_cov_image_block, stabilize_prior_cov_obs_mat
        ) 
from dataset.walnut import get_inner_block_indices

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

    avg_image_metric, avg_test_log_lik = [], []
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
        elif cfg.name == 'walnut':
            # pseudo-inverse computation
            trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
            U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
            # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
            S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
            # trafo_T_trafo_inv_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
            trafo_T_trafo_inv_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
            lik_hess_inv_diag_mean = np.mean(trafo_T_trafo_inv_diag) 
        print('noise_x_correction_term:', lik_hess_inv_diag_mean)

        if cfg.name in ['mnist', 'kmnist']:
            num_samples = 10000
            if cfg.baseline.name == 'mcdo':
                bayesianize_architecture(reconstructor.model, p=cfg.baseline.p)
                _, _ = reconstructor.reconstruct(
                    observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device), use_init_model=False, use_tv_loss=False)
                sample_recon = sample_from_bayesianized_model(reconstructor.model, filtbackproj.to(reconstructor.device), mc_samples=num_samples)
                mean = sample_recon.view(num_samples, -1).mean(dim=0)
                std = ( torch.var(sample_recon.view(num_samples, -1), dim=0) + lik_hess_inv_diag_mean) **.5
                log_prob_kernel_density = approx_kernel_density(example_image, sample_recon.cpu(), noise_x_correction_term=lik_hess_inv_diag_mean.cpu(), bw=cfg.baseline.bw) / example_image.numel()
            elif cfg.baseline.name == 'sgld':
                iterations = int(cfg.net.optim.iterations)
                cfg.net.optim.iterations = cfg.net.optim.iterations + (num_samples + 1) 
                _, _, sample_recon = reconstructor.reconstruct(
                    observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device), use_init_model=False, use_tv_loss=False, num_burn_in_steps=iterations)
                num_samples = sample_recon.shape[0]
                mean = sample_recon.view(num_samples, -1).mean(dim=0)
                std = ( torch.var(sample_recon.view(num_samples, -1), dim=0) + lik_hess_inv_diag_mean.cpu()) **.5
                log_prob_kernel_density = approx_kernel_density(example_image, sample_recon.cpu(), noise_x_correction_term=lik_hess_inv_diag_mean.cpu(), bw=cfg.baseline.bw) / example_image.numel()
                cfg.net.optim.iterations = iterations

            print('DIP reconstruction of sample {:d}'.format(i))
            print('PSNR:', PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('Test log-lik:', log_prob_kernel_density)

            data = {
                    'filtbackproj': filtbackproj.cpu().numpy(), 
                    'image': example_image.cpu().numpy(), 
                    'recon': mean.cpu().numpy(),
                    'std': std.cpu().numpy(), 
                    'psnr': PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()), 
                    'test_log_likelihood': log_prob_kernel_density
                    }

            np.savez('recon_info_{}'.format(i), **data)
            avg_image_metric.append(PSNR(mean.view(*example_image.shape[2:]).cpu().numpy(), example_image[0, 0].cpu().numpy()))
            avg_test_log_lik.append(log_prob_kernel_density)

        elif cfg.name == "walnut":

            num_samples = 16384
            if cfg.baseline.name == 'mcdo':
                if cfg.net.load_pretrain_model:
                    path = os.path.join(
                        get_original_cwd(),
                            cfg.net.learned_params_path if cfg.net.learned_params_path.endswith('.pt') \
                                else cfg.net.learned_params_path + '.pt')
                    reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
                    cfg.net.load_pretrain_model = False
                bayesianize_architecture(reconstructor.model, p=cfg.baseline.p)
                re_eval_bayes_dip_mcdo_baseline_load_path = cfg.get('re_eval_bayes_dip_mcdo_baseline_load_path', None)
                if re_eval_bayes_dip_mcdo_baseline_load_path:
                    path = os.path.join(re_eval_bayes_dip_mcdo_baseline_load_path, 'dip_model_{}.pt'.format(i))
                    print('loading mcdo baseline model for {} reconstruction from {}'.format(cfg.name, path))
                    reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
                else:
                    _, _ = reconstructor.reconstruct(
                        observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device), use_init_model=False, use_tv_loss=False)
                mc_sample_images = sample_from_bayesianized_model(reconstructor.model, filtbackproj.to(reconstructor.device), mc_samples=num_samples, device='cpu')
                recon = mc_sample_images.view(num_samples, -1).mean(dim=0).view(*example_image.shape)

            elif cfg.baseline.name == 'sgld':
                iterations = int(cfg.net.optim.iterations)
                cfg.net.optim.iterations = cfg.net.optim.iterations + (num_samples + 1) 
                _, _, mc_sample_images = reconstructor.reconstruct(
                    observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device), use_init_model=False, use_tv_loss=False, num_burn_in_steps=iterations)
                num_samples = mc_sample_images.shape[0]
                cfg.net.optim.iterations = iterations
                recon = mc_sample_images.view(num_samples, -1).mean(dim=0).view(*example_image.shape)

            block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

            errors = []
            block_diags = []
            block_log_probs = []
            block_mask_inds = []
            block_eps_values = []

            block_idx_list = cfg.density.compute_single_predictive_cov_block.block_idx
            if block_idx_list is None:
                block_idx_list = list(range(len(block_masks)))
            elif isinstance(block_idx_list, str):
                if block_idx_list == 'walnut_inner':
                    block_idx_list = get_inner_block_indices(block_size=cfg.density.block_size_for_approx)
                else:
                    raise ValueError('Unknown block_idx_list configuration: {}'.format(block_idx_list))
            else:
                try:
                    block_idx_list = list(block_idx_list)
                except TypeError:
                    block_idx_list = [block_idx_list]

            for block_idx in block_idx_list:

                print('starting with block', block_idx)

                mask = block_masks[block_idx]

                mc_sample_image_blocks = mc_sample_images.view(mc_sample_images.shape[0], -1)[:, mask]

                predictive_cov_image_block = approx_predictive_cov_image_block_from_samples(
                        mc_sample_image_blocks.to(reconstructor.device), noise_x_correction_term=lik_hess_inv_diag_mean)

                assert torch.all(torch.isfinite(predictive_cov_image_block))

                if cfg.density.do_eps_sweep:
                    eps_sweep_values = np.logspace(-7, -1, 13) * predictive_cov_image_block.diag().mean().item()
                    eps_sweep_block_log_probs = []
                    for eps_value in eps_sweep_values:
                        try:
                            block_log_prob_with_eps = predictive_image_block_log_prob(
                                    recon.to(reconstructor.device).flatten()[mask],
                                    example_image.to(reconstructor.device).flatten()[mask],
                                    predictive_cov_image_block + eps_value * torch.eye(predictive_cov_image_block.shape[0], device=predictive_cov_image_block.device))
                        except:
                            block_log_prob_with_eps = None
                        eps_sweep_block_log_probs.append(block_log_prob_with_eps)

                block_eps = stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode=cfg.density.eps_mode, eps=cfg.density.eps)
                
                block_log_prob = predictive_image_block_log_prob(
                        recon.to(reconstructor.device).flatten()[mask],
                        example_image.to(reconstructor.device).flatten()[mask],
                        predictive_cov_image_block)

                print('sample based log prob for block {}: {}'.format(block_idx, block_log_prob / mask.sum()))

                predictive_image_log_prob_block_dict = {'mask_inds': np.nonzero(mask)[0], 'block_log_prob': block_log_prob, 'block_diag': predictive_cov_image_block.diag(), 'block_eps': block_eps}
                if cfg.density.compute_single_predictive_cov_block.save_full_block:
                    predictive_image_log_prob_block_dict['block'] = predictive_cov_image_block
                if cfg.density.do_eps_sweep:
                    predictive_image_log_prob_block_dict['eps_sweep_values'] = eps_sweep_values
                    predictive_image_log_prob_block_dict['eps_sweep_block_log_probs'] = eps_sweep_block_log_probs
                   
                block_diags.append(predictive_image_log_prob_block_dict['block_diag'])
                block_mask_inds.append(np.nonzero(mask)[0])
                block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'])
                block_eps_values.append(predictive_image_log_prob_block_dict['block_eps'])

            approx_log_prob = torch.sum(torch.stack(block_log_probs))

            torch.save({'recon': recon, 'approx_log_prob': approx_log_prob, 'block_mask_inds': block_mask_inds, 'block_log_probs': block_log_probs, 'block_diags': block_diags, 'block_eps_values': block_eps_values},
                './predictive_image_log_prob_{}.pt'.format(i))

            print('approx log prob ', approx_log_prob / np.concatenate(block_mask_inds).flatten().shape[0])

        torch.save(reconstructor.model.state_dict(),
            './dip_model_{}.pt'.format(i))

    if cfg.name in ['kmnist', 'mnist']: 
        print('avg PSNR: ', np.mean(avg_image_metric))
        print('avg test log-lik: ', np.mean(avg_test_log_lik))
        overall_data = {
                    'avg_image_metric': np.mean(avg_image_metric), 
                    'avg_test_log_likelihood': np.mean(avg_test_log_lik)
                    }
        np.savez('overall_metrics{}'.format(i), **overall_data)


if __name__ == '__main__':
    coordinator()
