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
        get_image_block_masks)

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
            sample_dict = torch.load(os.path.join(load_path, 'sample_{}.pt'.format(i)), map_location=example_image.device)
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj)
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

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

        bayesianized_model.load_state_dict(torch.load(os.path.join(
                load_path, 'bayesianized_model_{}.pt'.format(i)),
                map_location=reconstructor.device))
        log_noise_model_variance_obs = torch.load(os.path.join(
                load_path, 'log_noise_model_variance_obs_{}.pt'.format(i)),
                map_location=reconstructor.device)['log_noise_model_variance_obs']

        cov_obs_mat = torch.load(os.path.join(load_path, 'cov_obs_mat_{}.pt'.format(i)), map_location=reconstructor.device)['cov_obs_mat']


        # compute predictive image log prob for block
        block_idx = cfg.density.compute_single_predictive_cov_block.block_idx

        block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)
        mask = block_masks[block_idx]

        predictive_cov_image_block = get_predictive_cov_image_block(
                mask, torch.linalg.cholesky(cov_obs_mat), ray_trafos, filtbackproj.to(reconstructor.device),
                bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules,
                vec_batch_size=cfg.mrglik.impl.vec_batch_size,
                eps=cfg.density.eps, cov_image_eps=cfg.density.cov_image_eps, return_cholesky=False)

        block_log_prob = predictive_image_block_log_prob(
                recon.to(reconstructor.device).flatten()[mask],
                example_image.to(reconstructor.device).flatten()[mask],
                predictive_cov_image_block)

        torch.save({'mask': mask, 'block_log_prob': block_log_prob, 'block_diag': predictive_cov_image_block.diag()},
            './predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i))

        print('block', block_idx, 'log prob ', block_log_prob / example_image.flatten()[mask].numel())


if __name__ == '__main__':
    coordinator()
