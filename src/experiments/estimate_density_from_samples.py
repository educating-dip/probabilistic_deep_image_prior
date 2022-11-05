import os
from itertools import islice
import numpy as np
from numpy import cov
import hydra
from omegaconf import OmegaConf, DictConfig
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
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, get_jac_fwAD_batch_ensemble,
        sample_from_posterior, predictive_image_block_log_prob, approx_predictive_cov_image_block_from_samples_batched, 
        predictive_image_block_log_prob_batched, get_image_block_mask_inds, stabilize_predictive_cov_image_block,
        stabilize_prior_cov_obs_mat, clamp_params, sample_from_posterior_via_jac, vec_weight_prior_cov_mul, 
        get_batched_jac_low_rank, get_reduced_model, get_inactive_and_leaf_modules_unet,
        get_prior_cov_obs_mat, get_prior_cov_obs_mat_jac_low_rank, get_cov_obs_low_rank_via_jac_low_rank,
        predictive_image_block_log_prob,
        get_cov_obs_low_rank, LowRankCovObsMat
        )
from dataset.walnut import get_inner_block_indices

### Estimate blocks of the predictive covariance matrix using samples.
### The samples are drawn based on the result from a `bayes_dip.py` run
### (specified via `density.compute_single_predictive_cov_block.load_path`);
### alternatively, the samples can be loaded from previous runs of this script
### (specified via `density.estimate_density_from_samples.samples_load_path_list`).

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

        if cfg.density.impl.reduce_model:
            inactive_modules, leaf_modules = get_inactive_and_leaf_modules_unet(model, keep_modules=modules)
            reduced_model, reduced_module_mapping = get_reduced_model(
                model, filtbackproj.to(reconstructor.device),
                replace_inactive=inactive_modules, replace_leaf=leaf_modules, return_module_mapping=True, share_parameters=True)
            modules = [reduced_module_mapping[m] for m in modules]
            assert all(m in reduced_model.modules() for m in modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            model = reduced_model

            fwAD_be_inactive_modules, fwAD_be_leaf_modules = get_inactive_and_leaf_modules_unet(fwAD_be_model, keep_modules=fwAD_be_modules)
            reduced_fwAD_be_model, reduced_fwAD_be_module_mapping = get_reduced_model(
                fwAD_be_model, torch.broadcast_to(filtbackproj.to(reconstructor.device), (cfg.mrglik.impl.vec_batch_size,) + filtbackproj.shape),
                replace_inactive=fwAD_be_inactive_modules, replace_leaf=fwAD_be_leaf_modules, return_module_mapping=True, share_parameters=True)
            fwAD_be_modules = [reduced_fwAD_be_module_mapping[m] for m in fwAD_be_modules]
            assert all(m in reduced_fwAD_be_model.modules() for m in fwAD_be_modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            fwAD_be_model = reduced_fwAD_be_model

        if cfg.density.estimate_density_from_samples.assemble_jac == 'full':
            if cfg.mrglik.impl.use_fwAD_for_jac_mat:
                jac = get_jac_fwAD_batch_ensemble(
                    filtbackproj.to(reconstructor.device),
                    fwAD_be_model,
                    fwAD_be_modules)
            else:
                jac = compute_jacobian_single_batch(
                    filtbackproj.to(reconstructor.device),
                    model, 
                    modules,
                    example_image.flatten().shape[0])
        elif cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank':
            if cfg.density.low_rank_jacobian.load_path is not None:
                jac_dict = torch.load(os.path.join(cfg.density.low_rank_jacobian.load_path, 'low_rank_jac.pt'), map_location=reconstructor.device)
                jac = (jac_dict['U'], jac_dict['S'], jac_dict['Vh'])
            else:
                low_rank_rank_dim = cfg.density.low_rank_jacobian.low_rank_dim + cfg.density.low_rank_jacobian.oversampling_param
                random_matrix = torch.randn(
                    (bayesianized_model.num_params_under_priors,
                            low_rank_rank_dim, 
                        ),
                    device=bayesianized_model.store_device
                    )
                add_batch_grad_hooks(model, modules)
                U, S, Vh = get_batched_jac_low_rank(random_matrix, filtbackproj.to(reconstructor.device),
                    bayesianized_model, model, fwAD_be_model, fwAD_be_modules,
                    cfg.mrglik.impl.vec_batch_size, cfg.density.low_rank_jacobian.low_rank_dim,
                    cfg.density.low_rank_jacobian.oversampling_param, 
                    use_cpu=cfg.density.low_rank_jacobian.use_cpu,
                    return_on_cpu=cfg.density.low_rank_jacobian.store_on_cpu
                    )
                jac = (U, S, Vh)
                if cfg.density.low_rank_jacobian.save:
                    torch.save({'U': U, 'S': S, 'Vh': Vh}, './low_rank_jac.pt')

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

        cov_obs_mat = None
        low_rank_cov_obs_mat = None 
        if cfg.density.estimate_density_from_samples.load_cov_obs_mat:  # user is responsible for checking that the cov_obs_mat is meaningful, then can activate this option
            if cfg.density.estimate_density_from_samples.low_rank_cov_obs_mat:
                low_rank_cov_obs_mat = LowRankCovObsMat(*torch.load(os.path.join(
                        cfg.density.compute_single_predictive_cov_block.cov_obs_mat_load_path, 'low_rank_cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['low_rank_cov_obs_mat'])
                low_rank_cov_obs_mat = (
                        LowRankCovObsMat(low_rank_cov_obs_mat.U.to(torch.float64), low_rank_cov_obs_mat.L.to(torch.float64), torch.tensor(float(low_rank_cov_obs_mat.log_noise_model_variance_obs), dtype=torch.float64), low_rank_cov_obs_mat.full_diag_eps) if cfg.use_double else
                        LowRankCovObsMat(low_rank_cov_obs_mat.U.to(torch.float32), low_rank_cov_obs_mat.L.to(torch.float32), torch.tensor(float(low_rank_cov_obs_mat.log_noise_model_variance_obs), dtype=torch.float64), low_rank_cov_obs_mat.full_diag_eps))
            else:
                cov_obs_mat_load_path = cfg.density.compute_single_predictive_cov_block.get('cov_obs_mat_load_path', None)
                if cov_obs_mat_load_path is None:
                    cov_obs_mat_load_path = load_path
                try:
                    cov_obs_mat = torch.load(os.path.join(cov_obs_mat_load_path, 'cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['cov_obs_mat'].detach()
                    cov_obs_mat = cov_obs_mat.to(torch.float64 if cfg.use_double else torch.float32)
                    print('loaded cov_obs_mat')
                except FileNotFoundError:
                    print('cov_obs_mat file not found')


        if not cfg.density.estimate_density_from_samples.low_rank_cov_obs_mat:
            if cov_obs_mat is not None:
                # note about potentially different prior selection
                cov_obs_mat_load_cfg = OmegaConf.load(os.path.join(cov_obs_mat_load_path, '.hydra', 'config.yaml'))
                if not (sorted(cfg.density.exclude_gp_priors_list) == sorted(cov_obs_mat_load_cfg.density.get('exclude_gp_priors_list', [])) and
                        sorted(cfg.density.exclude_normal_priors_list) == sorted(cov_obs_mat_load_cfg.density.get('exclude_normal_priors_list', []))):
                    print('note: prior selection seems to differ for the loaded cov_obs_mat')
            else:
                print('assembling cov_obs_mat')
                if cfg.density.estimate_density_from_samples.assemble_jac:
                    if scipy.sparse.issparse(ray_trafos['ray_trafo_mat']):
                        trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
                        if cfg.density.estimate_density_from_samples.assemble_jac == 'full':
                            jac_numpy = jac.cpu().numpy()
                            trafo = trafo.astype(jac_numpy.dtype)
                            jac_obs = torch.from_numpy(trafo @ jac_numpy).to(bayesianized_model.store_device)
                        elif cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank':
                            jac_U, jac_S, jac_Vh = jac
                            jac_U_numpy, jac_S_numpy, jac_Vh_numpy = jac_U.cpu().numpy(), jac_S.cpu().numpy(), jac_Vh.cpu().numpy()
                            jac_obs = None
                            if not cfg.density.low_rank_jacobian.use_closures_for_jac_obs:
                                trafo = trafo.astype(jac_U_numpy.dtype)
                                jac_obs = torch.from_numpy(trafo @ jac_U_numpy @ np.diag(jac_S_numpy) @ jac_Vh_numpy).to(bayesianized_model.store_device)
                    else:
                        trafo = torch.from_numpy(ray_trafos['ray_trafo_mat']).view(-1, np.prod(ray_trafos['space'].shape))
                        if cfg.density.estimate_density_from_samples.assemble_jac == 'full':
                            trafo = trafo.to(jac.dtype)
                            jac_obs = trafo.to(bayesianized_model.store_device) @ jac
                        elif cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank':
                            jac_U, jac_S, jac_Vh = jac
                            jac_obs = None
                            if not cfg.density.low_rank_jacobian.use_closures_for_jac_obs:
                                trafo = trafo.to(jac_U.dtype)
                                jac_obs = (trafo.to(jac_U.device) @ jac_U @ torch.diag_embed(jac_S) @ jac_Vh).to(bayesianized_model.store_device)
            
                    if not cfg.density.approx_inversion_using_conj_grad.use_conj_grad_inv:
                        # compute cov_obs_mat via assembled jac
                        if jac_obs is None:
                            assert cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank'
                            assert cfg.density.low_rank_jacobian.use_closures_for_jac_obs
                            cov_obs_mat = get_prior_cov_obs_mat_jac_low_rank(ray_trafos, bayesianized_model, jac,
                                log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, add_noise_model_variance_obs=True)
                        else:
                            cov_obs_mat = vec_weight_prior_cov_mul(bayesianized_model, jac_obs) @ jac_obs.transpose(1, 0)  # A * J * Sigma_theta * J.T * A.T
                            cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += log_noise_model_variance_obs.exp()
                    else:
                        reduced_rank_dim = cfg.density.low_rank_cov_obs_mat.low_rank_dim + cfg.density.low_rank_cov_obs_mat.oversampling_param
                        max_rank_dim = np.prod(ray_trafos['ray_trafo'].range.shape)
                        if reduced_rank_dim > max_rank_dim:
                            reduced_rank_dim = max_rank_dim
                            print('low_rank preconditioner: rank changed to full rank ({:d})'.format(reduced_rank_dim))
                        random_matrix_T = torch.randn(
                                (reduced_rank_dim, *ray_trafos['ray_trafo'].range.shape),
                                device=bayesianized_model.store_device
                                )
                        U, L, inv_cov_obs_approx = get_cov_obs_low_rank_via_jac_low_rank(
                            random_matrix_T, ray_trafos, filtbackproj.to(bayesianized_model.store_device),
                            bayesianized_model, jac,
                            log_noise_model_variance_obs.detach(),
                            cfg.mrglik.impl.vec_batch_size,
                            reduced_rank_dim - cfg.density.low_rank_cov_obs_mat.oversampling_param,
                            cfg.density.low_rank_cov_obs_mat.oversampling_param,
                            )
                        preconditioner = U, L, inv_cov_obs_approx
                else:
                    # compute cov_obs_mat via closure
                    if not cfg.density.approx_inversion_using_conj_grad.use_conj_grad_inv: 
                        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, model,
                            fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                            cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp, add_noise_model_variance_obs=True)
                    else:
                        reduced_rank_dim = cfg.density.low_rank_cov_obs_mat.low_rank_dim + cfg.density.low_rank_cov_obs_mat.oversampling_param
                        max_rank_dim = np.prod(ray_trafos['ray_trafo'].range.shape)
                        if reduced_rank_dim > max_rank_dim:
                            reduced_rank_dim = max_rank_dim
                            print('low_rank preconditioner: rank changed to full rank ({:d})'.format(reduced_rank_dim))
                        random_matrix_T = torch.randn(
                                (reduced_rank_dim, *ray_trafos['ray_trafo'].range.shape),
                                device=bayesianized_model.store_device
                                )
                        U, L, inv_cov_obs_approx = get_cov_obs_low_rank(
                            random_matrix_T, ray_trafos, filtbackproj.to(bayesianized_model.store_device),
                            bayesianized_model, model, fwAD_be_model, fwAD_be_modules,
                            log_noise_model_variance_obs.detach(),
                            cfg.mrglik.impl.vec_batch_size,
                            reduced_rank_dim - cfg.density.low_rank_cov_obs_mat.oversampling_param,
                            cfg.density.low_rank_cov_obs_mat.oversampling_param,
                            use_fwAD_for_jvp=True
                            )
                        preconditioner = U, L, inv_cov_obs_approx
        else:
            if low_rank_cov_obs_mat is not None:
                # note about potentially different prior selection
                low_rank_cov_obs_mat_load_cfg = OmegaConf.load(os.path.join(cfg.density.compute_single_predictive_cov_block.cov_obs_mat_load_path, '.hydra', 'config.yaml'))
                if not (sorted(cfg.density.exclude_gp_priors_list) == sorted(low_rank_cov_obs_mat_load_cfg.density.get('exclude_gp_priors_list', [])) and
                        sorted(cfg.density.exclude_normal_priors_list) == sorted(low_rank_cov_obs_mat_load_cfg.density.get('exclude_normal_priors_list', []))):
                    print('note: prior selection seems to differ for the loaded low_rank_cov_obs_mat')
            else:
                print('assembling low_rank_cov_obs_mat')
                random_matrix_T = torch.randn(
                    (cfg.density.low_rank_cov_obs_mat.low_rank_dim + cfg.density.low_rank_cov_obs_mat.oversampling_param,
                            *ray_trafos['ray_trafo'].range.shape),
                    device=bayesianized_model.store_device
                    )
                U, L = get_cov_obs_low_rank(random_matrix_T, ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, 
                        model, fwAD_be_model, fwAD_be_modules,  
                        log_noise_model_variance_obs,
                        cfg.mrglik.impl.vec_batch_size, cfg.density.low_rank_cov_obs_mat.low_rank_dim, 
                        cfg.density.low_rank_cov_obs_mat.oversampling_param, 
                        use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp,
                        use_cpu=False,
                        return_inverse=False
                        )
                low_rank_cov_obs_mat = LowRankCovObsMat(U, L, log_noise_model_variance_obs)
        
        if cfg.density.estimate_density_from_samples.save_cov_obs_mat:
            if cfg.density.estimate_density_from_samples.low_rank_cov_obs_mat:
                torch.save({'low_rank_cov_obs_mat': tuple(low_rank_cov_obs_mat)}, './low_rank_cov_obs_mat_{}.pt'.format(i))
            else:
                torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

        override_noise_model_variance_obs = cfg.density.compute_single_predictive_cov_block.get('override_noise_model_variance_obs', None)
        if override_noise_model_variance_obs is not None:
            if low_rank_cov_obs_mat is not None:
                low_rank_cov_obs_mat = LowRankCovObsMat(low_rank_cov_obs_mat.U, low_rank_cov_obs_mat.L, np.log(override_noise_model_variance_obs))
            if cov_obs_mat is not None: 
                orig_noise_model_variance_obs = np.exp(log_noise_model_variance_obs.item())
                log_noise_model_variance_obs.data[:] = np.log(override_noise_model_variance_obs)
                cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += (override_noise_model_variance_obs - orig_noise_model_variance_obs)  # adjust diagonal to overridden sigma_y

        # stabilize Kyy
        if low_rank_cov_obs_mat is not None:
            if cfg.density.cov_obs_mat_eps_mode == 'abs':
                low_rank_cov_obs_mat = LowRankCovObsMat(low_rank_cov_obs_mat.U, low_rank_cov_obs_mat.L, low_rank_cov_obs_mat.log_noise_model_variance_obs, cfg.density.cov_obs_mat_eps)
            else:
                raise NotImplementedError
        if cov_obs_mat is not None: 
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

        mc_sample_images = []
        samples_load_path_list = cfg.density.estimate_density_from_samples.samples_load_path_list
        if samples_load_path_list:
            if isinstance(samples_load_path_list, str):
                samples_load_path_list = [samples_load_path_list]
            for samples_load_path in samples_load_path_list:
                if not os.path.isdir(samples_load_path):
                    print('skipping non-existent directory:', samples_load_path)
                    continue
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
            conj_grad_kwards = None
            if not cfg.density.approx_inversion_using_conj_grad.use_conj_grad_inv:
                cov_obs_mat_chol_or_low_rank = torch.linalg.cholesky(cov_obs_mat) if not cfg.density.estimate_density_from_samples.low_rank_cov_obs_mat else low_rank_cov_obs_mat
            else: 
                cov_obs_mat_chol_or_low_rank = None
                conj_grad_kwards = {
                        'use_conj_grad_inv': True,
                        'max_cg_iter': cfg.density.approx_inversion_using_conj_grad.max_cg_iter,
                        'tolerance': cfg.density.approx_inversion_using_conj_grad.tolerance,
                        'preconditioner': preconditioner,
                        'log_noise_model_variance_obs': log_noise_model_variance_obs.detach(),
                }
            for chunk_idx, sample_idx in enumerate(range(0, cfg.density.num_mc_samples, cfg.density.estimate_density_from_samples.save_samples_chunk_size)):
                    print('drawing samples from posterior: {}/{}'.format(sample_idx, cfg.density.num_mc_samples))
                    chunk_size = min(cfg.density.estimate_density_from_samples.save_samples_chunk_size, cfg.density.num_mc_samples - sample_idx)
                    if cfg.density.estimate_density_from_samples.assemble_jac:
                        mc_sample_images_chunk, res_norm_list = sample_from_posterior_via_jac(ray_trafos, observation.to(reconstructor.device), jac, cov_obs_mat_chol_or_low_rank, bayesianized_model,
                                log_noise_model_variance_obs.detach(), 
                                mc_samples=chunk_size, vec_batch_size=cfg.mrglik.impl.vec_batch_size, device='cpu',
                                low_rank_jac=(cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank'),
                                conj_grad_kwards=conj_grad_kwards,
                        )
                        if res_norm_list:
                            torch.save(torch.cat(res_norm_list).view(-1).cpu(),
                                './residual_per_sample_via_jac.pt')              
                    else:
                        mc_sample_images_chunk, res_norm_list = sample_from_posterior(
                            ray_trafos, observation.to(reconstructor.device), filtbackproj.to(reconstructor.device),
                            cov_obs_mat_chol_or_low_rank, model, bayesianized_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs.detach(), 
                            mc_samples=chunk_size, vec_batch_size=cfg.mrglik.impl.vec_batch_size, device='cpu', conj_grad_kwards=conj_grad_kwards
                        )
                        if res_norm_list:
                            torch.save(torch.cat(res_norm_list).view(-1).cpu(),
                                './residual_per_sample_via_closure.pt')

                    mc_sample_images.append(mc_sample_images_chunk)
                    if cfg.density.estimate_density_from_samples.save_samples:
                        torch.save(mc_sample_images_chunk,
                                './posterior_samples_chunk{}_{}.pt'.format(chunk_idx, i))
        mc_sample_images = torch.cat(mc_sample_images, axis=0)
        limit_loaded_samples = cfg.density.estimate_density_from_samples.get('limit_loaded_samples', None)
        if limit_loaded_samples:
            mc_sample_images = mc_sample_images[:limit_loaded_samples]
        print('total number of posterior samples:', mc_sample_images.shape[0])

        # compute predictive image log prob for blocks

        if cfg.density.estimate_density_from_samples.save_single_result_file:
            block_mask_inds = {}
            block_log_probs = {}
            block_diags = {}
            block_eps_values = {}

        all_block_mask_inds = get_image_block_mask_inds(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

        block_idx_list = cfg.density.compute_single_predictive_cov_block.block_idx  # may be used to restrict to a subset of blocks
        if block_idx_list is None:
            block_idx_list = list(range(len(all_block_mask_inds)))
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

        errors = []
        for j in range(0, len(block_idx_list), cfg.density.estimate_density_from_samples.batch_size):
            batch_block_inds = block_idx_list[j:j+cfg.density.estimate_density_from_samples.batch_size]
            print('starting with blocks', batch_block_inds)

            batch_len_mask_inds = [len(all_block_mask_inds[block_idx]) for block_idx in batch_block_inds]
            max_len_mask_inds = max(batch_len_mask_inds)

            batch_mc_sample_image_blocks = torch.stack([
                    torch.nn.functional.pad(mc_sample_images.view(mc_sample_images.shape[0], -1)[:, all_block_mask_inds[block_idx]], (0, max_len_mask_inds - len_mask_inds))
                    for block_idx, len_mask_inds in zip(batch_block_inds, batch_len_mask_inds)])

            batch_predictive_cov_image_block = approx_predictive_cov_image_block_from_samples_batched(
                    batch_mc_sample_image_blocks.to(reconstructor.device), noise_x_correction_term=lik_hess_inv_diag_mean)
            # use identity for padding dims in predictive_cov_image_block
            # (the determinant then is the same as for predictive_cov_image_block[:len_mask_inds, :len_mask_inds])
            for k, len_mask_inds in enumerate(batch_len_mask_inds):
                batch_predictive_cov_image_block[k, np.arange(len_mask_inds, max_len_mask_inds), np.arange(len_mask_inds, max_len_mask_inds)] = 1.

            batch_invalid_values = torch.sum(torch.logical_not(torch.isfinite(batch_predictive_cov_image_block)), dim=(1, 2)) != 0
            batch_invalid_values_block_inds = torch.tensor(batch_block_inds)[batch_invalid_values].tolist()
            if len(batch_invalid_values_block_inds) > 0:
                print('skipping {:d} blocks due to nan or inf occurences'.format(len(batch_invalid_values_block_inds)))
                errors += batch_invalid_values_block_inds
                batch_valid_values = torch.logical_not(batch_invalid_values).cpu().numpy()
                # restrict batch to valid blocks
                batch_block_inds = np.asarray(batch_block_inds)[batch_valid_values].tolist()
                batch_mc_sample_image_blocks = batch_mc_sample_image_blocks[batch_valid_values]
                batch_predictive_cov_image_block = batch_predictive_cov_image_block[batch_valid_values]

            if len(batch_block_inds) == 0:
                continue  # no blocks left in the batch

            if cfg.density.do_eps_sweep:

                for k, block_idx in enumerate(batch_block_inds):
                    mask_inds = all_block_mask_inds[block_idx]
                    predictive_cov_image_block = batch_predictive_cov_image_block[k,:len(mask_inds),:len(mask_inds)]

                    eps_sweep_values = np.logspace(-7, -1, 13) * predictive_cov_image_block.diag().mean().item()
                    eps_sweep_block_log_probs = []
                    for eps_value in eps_sweep_values:
                        try:
                            block_log_prob_with_eps = predictive_image_block_log_prob(
                                    recon.to(reconstructor.device).flatten()[mask_inds],
                                    example_image.to(reconstructor.device).flatten()[mask_inds],
                                    predictive_cov_image_block + eps_value * torch.eye(predictive_cov_image_block.shape[0], device=predictive_cov_image_block.device))
                        except:
                            block_log_prob_with_eps = None
                        eps_sweep_block_log_probs.append(block_log_prob_with_eps)

            batch_not_stabilizable = []
            batch_not_stabilizable_block_inds = []

            batch_block_eps = []
            if cfg.density.eps:
                for k, block_idx in enumerate(batch_block_inds):
                    mask_inds = all_block_mask_inds[block_idx]
                    predictive_cov_image_block = batch_predictive_cov_image_block[k,:len(mask_inds),:len(mask_inds)]

                    try:
                        block_eps = stabilize_predictive_cov_image_block(predictive_cov_image_block, eps_mode=cfg.density.eps_mode, eps=cfg.density.eps)
                        batch_block_eps.append(block_eps)
                    except AssertionError:
                        batch_not_stabilizable.append(k)
                        batch_not_stabilizable_block_inds.append(block_idx)
                        batch_block_eps.append(np.nan)
            else:
                batch_block_eps = [0. for _ in batch_block_inds]

            if len(batch_not_stabilizable_block_inds) > 0:
                print('skipping {:d} blocks due to failed stabilizing attempts'.format(len(batch_not_stabilizable_block_inds)))
                errors += batch_not_stabilizable_block_inds
                batch_stabilizable = np.setdiff1d(range(len(batch_block_inds)), batch_not_stabilizable)
                # restrict batch to valid blocks
                batch_block_inds = np.asarray(batch_block_inds)[batch_stabilizable].tolist()
                batch_mc_sample_image_blocks = batch_mc_sample_image_blocks[batch_stabilizable]
                batch_predictive_cov_image_block = batch_predictive_cov_image_block[batch_stabilizable]            

            if len(batch_block_inds) == 0:
                continue  # no blocks left in the batch

            batch_recon = torch.stack([
                    torch.nn.functional.pad(recon.flatten()[all_block_mask_inds[block_idx]], (0, max_len_mask_inds - len_mask_inds))
                    for block_idx, len_mask_inds in zip(batch_block_inds, batch_len_mask_inds)])
            batch_example_image = torch.stack([
                    torch.nn.functional.pad(example_image.flatten()[all_block_mask_inds[block_idx]], (0, max_len_mask_inds - len_mask_inds))
                    for block_idx, len_mask_inds in zip(batch_block_inds, batch_len_mask_inds)])

            batch_block_log_prob = predictive_image_block_log_prob_batched(
                    batch_recon.to(reconstructor.device),
                    batch_example_image.to(reconstructor.device),
                    batch_predictive_cov_image_block)

            for k, block_idx in enumerate(batch_block_inds):
                mask_inds = all_block_mask_inds[block_idx]
                predictive_cov_image_block = batch_predictive_cov_image_block[k,:len(mask_inds),:len(mask_inds)]

                block_log_prob = batch_block_log_prob[k]

                print('sample based log prob for block {}: {}'.format(block_idx, block_log_prob / len(mask_inds)))

                predictive_image_log_prob_block_dict = {'mask_inds': mask_inds, 'block_log_prob': block_log_prob, 'block_diag': predictive_cov_image_block.diag(), 'block_eps': batch_block_eps[k]}
                if cfg.density.compute_single_predictive_cov_block.save_full_block:
                    predictive_image_log_prob_block_dict['block'] = predictive_cov_image_block
                if cfg.density.do_eps_sweep:
                    predictive_image_log_prob_block_dict['eps_sweep_values'] = eps_sweep_values
                    predictive_image_log_prob_block_dict['eps_sweep_block_log_probs'] = eps_sweep_block_log_probs

                if cfg.density.estimate_density_from_samples.save_block_files:
                    torch.save(predictive_image_log_prob_block_dict,
                        './predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i))

                if cfg.density.estimate_density_from_samples.save_single_result_file:
                    block_mask_inds[block_idx] = predictive_image_log_prob_block_dict['mask_inds']
                    block_log_probs[block_idx] = predictive_image_log_prob_block_dict['block_log_prob']
                    block_diags[block_idx] = predictive_image_log_prob_block_dict['block_diag']
                    block_eps_values[block_idx] = predictive_image_log_prob_block_dict['block_eps']

        if errors:
            print('errors occured in the following blocks:', errors)

        if cfg.density.estimate_density_from_samples.save_single_result_file:
            assert not errors
            approx_log_prob = torch.sum(torch.stack(list(block_log_probs.values())))
            torch.save({'approx_log_prob': approx_log_prob, 'block_mask_inds': list(block_mask_inds.values()), 'block_log_probs': list(block_log_probs.values()), 'block_diags': list(block_diags.values()), 'block_eps_values': list(block_eps_values.values())},
                './predictive_image_log_prob_{}.pt'.format(i))

        num_pixels_in_blocks = sum([len(mask_inds) for mask_inds in block_mask_inds.values()])

        print('approx log prob (mean over {} px): {}'.format(num_pixels_in_blocks, approx_log_prob / num_pixels_in_blocks))

if __name__ == '__main__':
    coordinator()
