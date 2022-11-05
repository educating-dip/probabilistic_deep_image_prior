import os
from itertools import islice
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import scipy.sparse
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, get_jac_fwAD_batch_ensemble,
        get_prior_cov_obs_mat, clamp_params,
        vec_weight_prior_cov_mul, get_batched_jac_low_rank, get_reduced_model, get_inactive_and_leaf_modules_unet, get_prior_cov_obs_mat_jac_low_rank,
)

### Assemble the covariance matrix in observation space based on the model and
### mrglik-optimization results of a `bayes_dip.py` run (specified via
### `density.assemble_cov_obs_mat.load_path`).
### By specifying different values for
### `density.assemble_cov_obs_mat.sub_slice_batches`, one can parallelize the
### computation and afterwards merge results using
### `merge_cov_obs_mat_sub_slices.py`.

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

    assert cfg.density.assemble_cov_obs_mat.load_path is not None, "no previous run path specified (density.assemble_cov_obs_mat.load_path)"

    load_path = cfg.density.assemble_cov_obs_mat.load_path
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

        load_iter = cfg.density.assemble_cov_obs_mat.get('load_mrglik_opt_iter', None)
        missing_keys, _ = bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i) if load_iter is None else 'bayesianized_model_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device), strict=False)
        assert not missing_keys
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i) if load_iter is None else 'log_noise_model_variance_obs_mrglik_opt_recon_num_{}_iter_{}.pt'.format(i, load_iter)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization
            clamp_params(bayesianized_model.gp_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)
            clamp_params(bayesianized_model.normal_log_variances, min=cfg.mrglik.priors.clamp_variances_min_log)

        sub_slice_batches = cfg.density.assemble_cov_obs_mat.get('sub_slice_batches', None)
        if sub_slice_batches is not None:
            sub_slice_batches = slice(*sub_slice_batches)

        if cfg.density.estimate_density_from_samples.assemble_jac:
            # compute cov_obs_mat via jac
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
            if jac_obs is None:
                assert cfg.density.estimate_density_from_samples.assemble_jac == 'low_rank'
                assert cfg.density.low_rank_jacobian.use_closures_for_jac_obs
                cov_obs_mat = get_prior_cov_obs_mat_jac_low_rank(ray_trafos, bayesianized_model, jac,
                    log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, add_noise_model_variance_obs=True, sub_slice_batches=sub_slice_batches)
            else:
                assert sub_slice_batches is None, 'sub-slicing is not supported (would not be useful, jac_obs can be assembled)'
                cov_obs_mat = vec_weight_prior_cov_mul(bayesianized_model, jac_obs) @ jac_obs.transpose(1, 0)  # A * J * Sigma_theta * J.T * A.T
                cov_obs_mat[np.diag_indices(cov_obs_mat.shape[0])] += log_noise_model_variance_obs.exp()
        else:
            cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, model,
                    fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                    cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp, add_noise_model_variance_obs=True, sub_slice_batches=sub_slice_batches)

        if sub_slice_batches is not None:
            torch.save({'cov_obs_mat_sub_slice': cov_obs_mat, 'sub_slice_batches': sub_slice_batches}, './cov_obs_mat_sub_slice_{}.pt'.format(i))
        else:
            torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

if __name__ == '__main__':
    coordinator()
