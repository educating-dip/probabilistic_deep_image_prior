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
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        get_predictive_cov_image_block, predictive_image_block_log_prob,
        get_image_block_masks, get_prior_cov_obs_mat, clamp_params)

### Compute a single block
### (specified via `density.assemble_cov_obs_mat.block_idx`) of
### the predictive covariance matrix based on the model and mrglik-optimization
### results of a previous run (specified via
### `density.assemble_cov_obs_mat.load_path`).
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

    assert cfg.density.assemble_cov_obs_mat.load_path is not None, "no previous run path specified (density.assemble_cov_obs_mat.load_path)"

    load_path = cfg.density.assemble_cov_obs_mat.load_path

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

        bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i)),
                map_location=reconstructor.device))
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        if cfg.mrglik.priors.clamp_variances:  # this only has an effect if clamping was turned off during optimization
            clamp_params(bayesianized_model.gp_log_variances, min=-4.5)
            clamp_params(bayesianized_model.normal_log_variances, min=-4.5)

        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model,
                fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=cfg.mrglik.impl.use_fwAD_for_jvp, add_noise_model_variance_obs=True)

        torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

if __name__ == '__main__':
    coordinator()
