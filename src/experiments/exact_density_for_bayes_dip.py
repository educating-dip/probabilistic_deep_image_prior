import os
from itertools import islice
import numpy as np
import random
import hydra
from omegaconf import OmegaConf, DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
from dataset import extract_trafos_as_matrices
import torch
import scipy
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel, BlocksGPpriors
from linearized_laplace import compute_jacobian_single_batch, image_space_lin_model_post_pred_cov, gaussian_log_prob
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        sample_from_posterior, approx_predictive_cov_image_block_from_samples, predictive_image_block_log_prob,
        get_image_block_masks, stabilize_predictive_cov_image_block,
        stabilize_prior_cov_obs_mat, clamp_params)

### Compute exact predictive covariance matrix (blocks) based on the model and
### mrglik-optimization results of a `bayes_dip.py` run (specified via
### `density.compute_single_predictive_cov_block.load_path`).

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

    assert load_cfg.beam_num_angle == cfg.beam_num_angle
    assert load_cfg.noise_specs.stddev == cfg.noise_specs.stddev
    assert load_cfg.use_double == cfg.use_double

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
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj, atol=1e-7)
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
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

        # estimate the Jacobian
        Jac = compute_jacobian_single_batch(
            filtbackproj.to(reconstructor.device),
            reconstructor.model, 
            bayesianized_model.get_all_modules_under_prior(),
            example_image.flatten().shape[0]
            )
        trafos = extract_trafos_as_matrices(ray_trafos)
        trafo = trafos[0]
        if cfg.use_double:
            trafo = trafo.to(torch.float64)
        proj_recon = trafo @ recon.flatten()
        Jac_obs = trafo.to(reconstructor.device) @ Jac

        recon = torch.from_numpy(recon[None, None])
        if load_cfg.linearize_weights:
            linearized_weights_dict = torch.load(os.path.join(load_path, 'linearized_weights_{}.pt'.format(i)), map_location=reconstructor.device)
            linearized_weights = linearized_weights_dict['linearized_weights']
            lin_pred = linearized_weights_dict['linearized_prediction']

            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))

        else:
            linearized_weights = None
            lin_pred = None

        block_priors = BlocksGPpriors(
            reconstructor.model,
            bayesianized_model,
            reconstructor.device,
            cfg.mrglik.priors.lengthscale_init,
            cfg.mrglik.priors.variance_init,
            lin_weights=linearized_weights
            )

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
            clamp_params(bayesianized_model.gp_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)
            clamp_params(bayesianized_model.normal_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)

        lik_hess_inv_diag_mean = None
        if cfg.name in ['mnist', 'kmnist']:
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

        # like in low_rank_obs_space_mrg_lik_opt
        (_, model_post_cov_no_predcp, Kxx_no_predcp) = image_space_lin_model_post_pred_cov(
            block_priors,
            Jac,
            Jac_obs, 
            torch.exp(log_noise_model_variance_obs)
            )

        # compute predictive image log prob for blocks
        block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

        block_idx_list = cfg.density.compute_single_predictive_cov_block.block_idx  # may be used to restrict to a subset of blocks
        if block_idx_list is None:
            block_idx_list = list(range(len(block_masks)))
        else:
            try:
                block_idx_list = list(block_idx_list)
            except TypeError:
                block_idx_list = [block_idx_list]

        block_diags = []
        block_log_probs = []
        block_mask_inds = []
        for block_idx in block_idx_list:
            print('starting with block', block_idx)

            mask = block_masks[block_idx]

            predictive_cov_image_block = model_post_cov_no_predcp[mask][:, mask]

            assert torch.all(torch.isfinite(predictive_cov_image_block))

            # like in low_rank_obs_space_mrg_lik_opt, but for a single block
            # computing test-loglik MLL
            block_log_prob = gaussian_log_prob(
                example_image.flatten().to(block_priors.store_device)[mask],
                recon.flatten().to(block_priors.store_device)[mask],
                predictive_cov_image_block,
                lik_hess_inv_diag_mean * torch.eye(mask.sum(), device=block_priors.store_device)
                )
            block_log_prob = block_log_prob * mask.sum()  # for consistency with other block-based code, we undo the division in gaussian_log_prob

            print('sample based log prob for block {}: {}'.format(block_idx, block_log_prob / mask.sum()))

            predictive_image_log_prob_block_dict = {'mask_inds': np.nonzero(mask)[0], 'block_log_prob': block_log_prob, 'block_diag': predictive_cov_image_block.diag()}
            if cfg.density.compute_single_predictive_cov_block.save_full_block:
                predictive_image_log_prob_block_dict['block'] = predictive_cov_image_block

            block_diags.append(predictive_image_log_prob_block_dict['block_diag'])
            block_mask_inds.append(np.nonzero(mask)[0])
            block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'])

        approx_log_prob = torch.sum(torch.stack(block_log_probs))

        torch.save({'approx_log_prob': approx_log_prob, 'block_mask_inds': block_mask_inds, 'block_log_probs': block_log_probs, 'block_diags': block_diags},
            './predictive_image_log_prob_{}.pt'.format(i))

        print('approx log prob ', approx_log_prob / np.concatenate(block_mask_inds).flatten().shape[0])

if __name__ == '__main__':
    coordinator()
