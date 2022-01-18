import os
from itertools import islice
import numpy as np
import random
import hydra
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import scipy
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        sample_from_posterior, approx_predictive_cov_image_block_from_samples, predictive_image_block_log_prob,
        get_image_block_masks, stabilize_predictive_cov_image_block,
        stabilize_prior_cov_obs_mat, clamp_params)

### Compute a single block
### (specified via `density.compute_single_predictive_cov_block.block_idx`) of
### the predictive covariance matrix based on the model and mrglik-optimization
### results of a previous run (specified via
### `density.compute_single_predictive_cov_block.load_path`).
### This allows to parallelize with multiple jobs; after all jobs are finished,
### the approx. predictive log prob can be evaluated with
### ``merge_single_block_predictive_image_log_probs.py``.

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
    assert cfg.density.compute_single_predictive_cov_block.block_idx is not None, "no block index specified (density.compute_single_predictive_cov_block.block_idx)"

    load_path = cfg.density.compute_single_predictive_cov_block.load_path

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
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj)
            # filtbackproj = sample_dict['filtbackproj']
            # observation = sample_dict['observation']
            # example_image = sample_dict['ground_truth']
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
            path = os.path.join(load_path, 'dip_model_{}.pt'.format(i))
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
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

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
        
        modules = bayesianized_model.get_all_modules_under_prior()
        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, use_copy='share_parameters')
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

        load_iter = cfg.density.compute_single_predictive_cov_block.get('load_mrglik_opt_iter', None)
        bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i) if load_iter is None else 'bayesianized_model_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device))
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i) if load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']
        override_noise_model_variance_obs = cfg.density.compute_single_predictive_cov_block.get('override_noise_model_variance_obs', None)
        if override_noise_model_variance_obs is not None:
            orig_noise_model_variance_obs = np.exp(log_noise_model_variance_obs.item())
            log_noise_model_variance_obs.data[:] = np.log(override_noise_model_variance_obs)

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization; if post-hoc clamping, we expect the user to load a cov_obs_mat that was computed with clamping, too
            clamp_params(bayesianized_model.gp_log_variances, min=-4.5)
            clamp_params(bayesianized_model.normal_log_variances, min=-4.5)

        cov_obs_mat_load_path = cfg.density.compute_single_predictive_cov_block.get('cov_obs_mat_load_path', None)
        if cov_obs_mat_load_path is None:
            cov_obs_mat_load_path = load_path
        cov_obs_mat = torch.load(os.path.join(cov_obs_mat_load_path, 'cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['cov_obs_mat'].detach()
        cov_obs_mat = cov_obs_mat.to(torch.float64 if cfg.use_double else torch.float32)
        if override_noise_model_variance_obs is not None:
            cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += (override_noise_model_variance_obs - orig_noise_model_variance_obs)  # adjust diagonal to overridden sigma_y

        cov_obs_mat = 0.5 * (cov_obs_mat + cov_obs_mat.T)  # in case of numerical issues leading to asymmetry
        stabilize_prior_cov_obs_mat(cov_obs_mat, eps_mode=cfg.density.cov_obs_mat_eps_mode, eps=cfg.density.cov_obs_mat_eps)

        lik_hess_inv_diag_mean = None
        if cfg.name in ['mnist', 'kmnist']:
            from dataset import extract_trafos_as_matrices
            import tensorly as tl
            tl.set_backend('pytorch')
            # pseudo-inverse computation
            trafos = extract_trafos_as_matrices(ray_trafos)
            trafo = trafos[0]
            if cfg.use_double:
                trafo = trafo.to(torch.float64)
            trafo = trafo.to(reconstructor.device)
            trafo_T_trafo = trafo.T @ trafo
            U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
            lik_hess_inv_diag_mean = (Vh.T @ torch.diag(1/S) @ U.T * torch.exp(log_noise_model_variance_obs)).diag().mean()
        elif cfg.name == 'walnut':
            # pseudo-inverse computation
            trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
            if cfg.use_double:
                trafo = trafo.astype(np.float64)
            U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
            # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
            S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
            # trafo_T_trafo_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
            trafo_T_trafo_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
            lik_hess_inv_diag_mean = np.mean(trafo_T_trafo_diag) * np.exp(log_noise_model_variance_obs.item())
        print('noise_x_correction_term:', lik_hess_inv_diag_mean)

        # draw samples from posterior
        if cfg.density.estimate_density_from_samples.seed is not None:
            torch.manual_seed(cfg.density.estimate_density_from_samples.seed)

        mc_sample_images = []
        if cfg.density.estimate_density_from_samples.samples_load_path_list:
            for samples_load_path in cfg.density.estimate_density_from_samples.samples_load_path_list:
                chunk_idx = 0
                finding_files = True
                while finding_files:
                    try:
                        mc_sample_images_chunk = torch.load(
                                os.path.join(samples_load_path, 'posterior_samples_chunk{}_{}.pt'.format(chunk_idx, i)), map_location='cpu')
                        mc_sample_images.append(mc_sample_images_chunk)
                        chunk_idx += 1
                    except FileNotFoundError:
                        finding_files = False
                print('loaded {} posterior sample chunks from path {}'.format(chunk_idx, samples_load_path))
        else:
            save_samples_num_chunks = cfg.density.estimate_density_from_samples.save_samples_chunk_size
            for chunk_idx, sample_idx in enumerate(range(0, cfg.density.num_mc_samples, cfg.density.estimate_density_from_samples.save_samples_chunk_size)):
                    print('drawing samples from posterior: {}/{}'.format(sample_idx, cfg.density.num_mc_samples))
                    chunk_size = min(cfg.density.estimate_density_from_samples.save_samples_chunk_size, cfg.density.num_mc_samples - sample_idx)
                    mc_sample_images_chunk = sample_from_posterior(ray_trafos, observation.to(reconstructor.device), filtbackproj.to(reconstructor.device),
                            torch.linalg.cholesky(cov_obs_mat), reconstructor.model, bayesianized_model, fwAD_be_model, fwAD_be_modules,
                            mc_samples=chunk_size, vec_batch_size=cfg.mrglik.impl.vec_batch_size, device='cpu')
                    mc_sample_images.append(mc_sample_images_chunk)
                    if cfg.density.estimate_density_from_samples.save_samples:
                        torch.save(mc_sample_images_chunk,
                                './posterior_samples_chunk{}_{}.pt'.format(chunk_idx, i))
        mc_sample_images = torch.cat(mc_sample_images, axis=0)
        print('total number of posterior samples:', mc_sample_images.shape[0])

        # compute predictive image log prob for blocks
        block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

        block_idx_list = cfg.density.compute_single_predictive_cov_block.block_idx
        try:
            block_idx_list = list(block_idx_list)
        except TypeError:
            block_idx_list = [block_idx_list]

        errors = []
        for block_idx in block_idx_list:
            print('starting with block', block_idx)

            mask = block_masks[block_idx]

            mc_sample_image_blocks = mc_sample_images.view(mc_sample_images.shape[0], -1)[:, mask]

            predictive_cov_image_block = approx_predictive_cov_image_block_from_samples(
                    mc_sample_image_blocks.to(reconstructor.device), noise_x_correction_term=lik_hess_inv_diag_mean)

            if not torch.all(torch.isfinite(predictive_cov_image_block)):
                errors.append(block_idx)
                print('skipping block due to nan or inf occurences')
                continue

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

            try:
                block_eps = stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode=cfg.density.eps_mode, eps=cfg.density.eps)
            except AssertionError:
                errors.append(block_idx)
                print('skipping block due to failed stabilizing attempt')
                continue

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

            torch.save(predictive_image_log_prob_block_dict,
                './predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i))

        if errors:
            print('errors occured in the following blocks:', errors)

if __name__ == '__main__':
    coordinator()
