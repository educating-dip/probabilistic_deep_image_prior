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
from dataset.walnut import get_inner_block_indices

def _get_reduced_be_model_and_modules_for_prior(filtbackproj, bayesianized_model, be_model, model_to_be_module_mapping, prior_kind, rel_prior_idx, vec_batch_size):
    assert prior_kind in ['gp', 'normal']
    modules_under_prior = (bayesianized_model.ref_modules_under_gp_priors if prior_kind == 'gp' else bayesianized_model.ref_modules_under_normal_priors)[rel_prior_idx]
    be_modules_under_prior = [model_to_be_module_mapping[m] for m in modules_under_prior]
    be_inactive_modules, be_leaf_modules = get_inactive_and_leaf_modules_unet(be_model, keep_modules=be_modules_under_prior)
    reduced_be_model, reduced_be_module_mapping = get_reduced_model(
        be_model, torch.broadcast_to(filtbackproj.to(bayesianized_model.store_device), (vec_batch_size,) + filtbackproj.shape),
        replace_inactive=be_inactive_modules, replace_leaf=be_leaf_modules, return_module_mapping=True, share_parameters=True)
    be_modules_under_prior = [reduced_be_module_mapping[m] for m in be_modules_under_prior]
    assert all(m in reduced_be_model.modules() for m in be_modules_under_prior), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
    return reduced_be_model, be_modules_under_prior

def sample_from_image_cov_priors(hooked_model, filtbackproj, bayesianized_model, be_model, be_modules, model_to_be_module_mapping, mc_samples, vec_batch_size, return_on_cpu=False):
    block_priors = BlocksGPpriors(
        hooked_model,
        bayesianized_model,
        bayesianized_model.store_device,
        0.1, 1)
    idx_params = block_priors.get_idx_parameters_per_block()
    idx_params_gp = idx_params[:len(bayesianized_model.gp_priors)]
    idx_params_normal = idx_params[len(bayesianized_model.gp_priors):]
    num_batches = ceil(mc_samples / vec_batch_size)
    mc_samples = num_batches * vec_batch_size
    s_images = []
    s_images_gp_priors = [[] for _ in bayesianized_model.gp_priors]
    s_images_normal_priors = [[] for _ in bayesianized_model.normal_priors]
    reduced_be_models_and_modules_gp_priors = [
        _get_reduced_be_model_and_modules_for_prior(filtbackproj, bayesianized_model, be_model, model_to_be_module_mapping, 'gp', j, vec_batch_size)
        for j, _ in enumerate(bayesianized_model.gp_priors)]
    reduced_be_models_and_modules_normal_priors = [
        _get_reduced_be_model_and_modules_for_prior(filtbackproj, bayesianized_model, be_model, model_to_be_module_mapping, 'normal', j, vec_batch_size)
        for j, _ in enumerate(bayesianized_model.normal_priors)]
    for i in tqdm(range(num_batches), desc='sample_from_image_cov'):
        sample_weight_vec_batch = _sample_from_prior_over_weights(bayesianized_model, vec_batch_size)
        # full network
        s_image = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec_batch, be_modules)
        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        s_images.append(s_image.cpu() if return_on_cpu else s_image)
        # individual Gaussian priors
        for j, _ in enumerate(bayesianized_model.gp_priors):
            # fwAD_JvP_batch_ensemble expects vectors containing values for the specified modules only
            sample_weight_vec_batch_prior = sample_weight_vec_batch[:, idx_params_gp[j][0]:idx_params_gp[j][1]]
            s_image_prior = fwAD_JvP_batch_ensemble(filtbackproj, reduced_be_models_and_modules_gp_priors[j][0], sample_weight_vec_batch_prior, reduced_be_models_and_modules_gp_priors[j][1])
            # sample_weight_vec_batch_prior = sample_weight_vec_batch.clone()
            # sample_weight_vec_batch_prior[:, :idx_params_gp[j][0]] = 0.
            # sample_weight_vec_batch_prior[:, idx_params_gp[j][1]:] = 0.
            # s_image_prior = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec_batch_prior, be_modules)
            s_image_prior = s_image_prior.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
            s_images_gp_priors[j].append(s_image_prior.cpu() if return_on_cpu else s_image_prior)
        # individual Normal priors
        for j, _ in enumerate(bayesianized_model.normal_priors):
            # fwAD_JvP_batch_ensemble expects vectors containing values for the specified modules only
            sample_weight_vec_batch_prior = sample_weight_vec_batch[:, idx_params_normal[j][0]:idx_params_normal[j][1]]
            s_image_prior = fwAD_JvP_batch_ensemble(filtbackproj, reduced_be_models_and_modules_normal_priors[j][0], sample_weight_vec_batch_prior, reduced_be_models_and_modules_normal_priors[j][1])
            # sample_weight_vec_batch_prior = sample_weight_vec_batch.clone()
            # sample_weight_vec_batch_prior[:, :idx_params_normal[j][0]] = 0.
            # sample_weight_vec_batch_prior[:, idx_params_normal[j][1]:] = 0.
            # s_image_prior = fwAD_JvP_batch_ensemble(filtbackproj, be_model, sample_weight_vec_batch_prior, be_modules)
            s_image_prior = s_image_prior.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
            s_images_normal_priors[j].append(s_image_prior.cpu() if return_on_cpu else s_image_prior)
    s_images = torch.cat(s_images, axis=0)
    s_images_gp_priors = [torch.cat(s_images_prior, axis=0) for s_images_prior in s_images_gp_priors]
    s_images_normal_priors = [torch.cat(s_images_prior, axis=0) for s_images_prior in s_images_normal_priors]
    return s_images, s_images_gp_priors, s_images_normal_priors

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

        model = reconstructor.model
        modules = bayesianized_model.get_all_modules_under_prior()

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        model_to_fwAD_be_module_mapping = {m: fwAD_be_module_mapping[be_module_mapping[m]] for m in modules}

        if 'batch_grad_hooks' not in model.__dict__:
            add_batch_grad_hooks(model, modules)

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

        load_iter = cfg.density.compute_single_predictive_cov_block.get('load_mrglik_opt_iter', None)
        missing_keys, _ = bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i) if load_iter is None else 'bayesianized_model_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device), strict=False)
        assert not missing_keys
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i) if load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization; if post-hoc clamping, we expect the user to load a cov_obs_mat that was computed with clamping, too
            clamp_params(bayesianized_model.gp_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)
            clamp_params(bayesianized_model.normal_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)

        cov_obs_mat_load_path = cfg.density.compute_single_predictive_cov_block.get('cov_obs_mat_load_path', None)
        if cov_obs_mat_load_path is None:
            cov_obs_mat_load_path = load_path
        cov_obs_mat = torch.load(os.path.join(cov_obs_mat_load_path, 'cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['cov_obs_mat'].detach()
        cov_obs_mat = cov_obs_mat.to(torch.float64 if cfg.use_double else torch.float32)
        print('loaded cov_obs_mat')

        if cfg.density.estimate_density_from_samples.save_cov_obs_mat:
            torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

        override_noise_model_variance_obs = cfg.density.compute_single_predictive_cov_block.get('override_noise_model_variance_obs', None)
        if override_noise_model_variance_obs is not None:
            orig_noise_model_variance_obs = np.exp(log_noise_model_variance_obs.item())
            log_noise_model_variance_obs.data[:] = np.log(override_noise_model_variance_obs)
            cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += (override_noise_model_variance_obs - orig_noise_model_variance_obs)  # adjust diagonal to overridden sigma_y

        cov_obs_mat = 0.5 * (cov_obs_mat + cov_obs_mat.T)  # in case of numerical issues leading to asymmetry
        stabilize_prior_cov_obs_mat(cov_obs_mat, eps_mode=cfg.density.cov_obs_mat_eps_mode, eps=cfg.density.cov_obs_mat_eps, eps_min_for_auto=cfg.density.cov_obs_mat_eps_min_for_auto)

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

        # adapted from https://stackoverflow.com/a/56407442
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        def update_mean_m2_(new_total_count, mean, M2, newValues):
            # newvalues - oldMean
            delta = newValues - mean
            mean += torch.sum(delta, dim=0) / new_total_count
            # newvalues - newMeant
            delta2 = newValues - mean
            M2 += torch.sum(delta * delta2, dim=0)

            return mean, M2

        # all_s_images = []
        # all_s_images_gp_priors = [[] for _ in enumerate(bayesianized_model.gp_priors)]
        # all_s_images_normal_priors = [[] for _ in enumerate(bayesianized_model.normal_priors)]

        count = 0
        mean, M2 = torch.zeros_like(filtbackproj), torch.zeros_like(filtbackproj)
        mean_gp_priors, M2_gp_priors = (
            [torch.zeros_like(filtbackproj) for  _ in enumerate(bayesianized_model.gp_priors)],
            [torch.zeros_like(filtbackproj) for  _ in enumerate(bayesianized_model.gp_priors)])
        mean_normal_priors, M2_normal_priors = (
            [torch.zeros_like(filtbackproj) for  _ in enumerate(bayesianized_model.normal_priors)],
            [torch.zeros_like(filtbackproj) for  _ in enumerate(bayesianized_model.normal_priors)])

        def save(filename, count, mean, M2, mean_gp_priors, M2_gp_priors, mean_normal_priors, M2_normal_priors):
            torch.save({
                'mean': mean,
                'var': M2 / count,
                'mean_gp_priors': mean_gp_priors,
                'var_gp_priors': [M2_prior / count for M2_prior in M2_gp_priors],
                'mean_normal_priors': mean_normal_priors,
                'var_normal_priors': [M2_prior / count for M2_prior in M2_normal_priors],
                'count': count,
            }, filename)

        batch_size = 128
        num_mc_samples = ceil(8192 / batch_size) * batch_size
        for k in range(0, num_mc_samples, batch_size):
            print('count: {:d}'.format(count))
            s_images, s_images_gp_priors, s_images_normal_priors = sample_from_image_cov_priors(reconstructor.model, filtbackproj.to(reconstructor.device), bayesianized_model, fwAD_be_model, fwAD_be_modules, model_to_fwAD_be_module_mapping, batch_size, cfg.mrglik.impl.vec_batch_size, return_on_cpu=True)

            # all_s_images.append(s_images)
            # for j, _ in enumerate(bayesianized_model.gp_priors):
            #     all_s_images_gp_priors[j].append(s_images_gp_priors[j])
            # for j, _ in enumerate(bayesianized_model.normal_priors):
            #     all_s_images_normal_priors[j].append(s_images_normal_priors[j])

            count += batch_size
            update_mean_m2_(count, mean, M2, s_images)
            for j, _ in enumerate(bayesianized_model.gp_priors):
                update_mean_m2_(count, mean_gp_priors[j], M2_gp_priors[j], s_images_gp_priors[j])
            for j, _ in enumerate(bayesianized_model.normal_priors):
                update_mean_m2_(count, mean_normal_priors[j], M2_normal_priors[j], s_images_normal_priors[j])

            save('mean_var_image_cov_priors_{:d}samples.pt'.format(count), count, mean, M2, mean_gp_priors, M2_gp_priors, mean_normal_priors, M2_normal_priors)

        # all_s_images = torch.cat(all_s_images, dim=0)
        # all_s_images_gp_priors = [torch.cat(all_s_images_prior, dim=0) for all_s_images_prior in all_s_images_gp_priors]
        # all_s_images_normal_priors = [torch.cat(all_s_images_prior, dim=0) for all_s_images_prior in all_s_images_normal_priors]

        # var_via_all = torch.var(all_s_images, dim=0, unbiased=False)
        # var_gp_priors_via_all = [torch.var(all_s_images_prior, dim=0, unbiased=False) for all_s_images_prior in all_s_images_gp_priors]
        # var_normal_priors_via_all = [torch.var(all_s_images_prior, dim=0, unbiased=False) for all_s_images_prior in all_s_images_normal_priors]

        # print('[via all] mean variance (full jacobian):', torch.mean(var_via_all).item())
        # print('[via all] max variance (full jacobian):', torch.max(var_via_all).item())
        # for j, var_prior_via_all in enumerate(var_gp_priors_via_all):
        #     print('[via all] mean variance (GP {:d}):'.format(j), torch.mean(var_prior_via_all).item())
        #     print('[via all] max variance (GP {:d}):'.format(j), torch.max(var_prior_via_all).item())
        # for j, var_prior_via_all in enumerate(var_normal_priors_via_all):
        #     print('[via all] mean variance (Normal {:d}):'.format(j), torch.mean(var_prior_via_all).item())
        #     print('[via all] max variance (Normal {:d}):'.format(j), torch.max(var_prior_via_all).item())

        save('mean_var_image_cov_priors.pt', num_mc_samples, mean, M2, mean_gp_priors, M2_gp_priors, mean_normal_priors, M2_normal_priors)

        var = M2 / count
        var_gp_priors = [M2_prior / count for M2_prior in M2_gp_priors]
        var_normal_priors = [M2_prior / count for M2_prior in M2_normal_priors]

        print('mean variance (full jacobian):', torch.mean(var).item())
        print('max variance (full jacobian):', torch.max(var).item())
        for j, var_prior in enumerate(var_gp_priors):
            print('mean variance (GP {:d}):'.format(j), torch.mean(var_prior).item())
            print('max variance (GP {:d}):'.format(j), torch.max(var_prior).item())
        for j, var_prior in enumerate(var_normal_priors):
            print('mean variance (Normal {:d}):'.format(j), torch.mean(var_prior).item())
            print('max variance (Normal {:d}):'.format(j), torch.max(var_prior).item())

if __name__ == '__main__':
    coordinator()
