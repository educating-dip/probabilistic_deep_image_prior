import os
from itertools import islice
import numpy as np
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
from linearized_weights import weights_linearization
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        optim_marginal_lik_low_rank, predictive_image_log_prob)

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

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
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

        if cfg.name in ['mnist', 'kmnist']:  
            recon, _ = reconstructor.reconstruct(
                observation, fbp=filtbackproj, ground_truth=example_image)
            torch.save(reconstructor.model.state_dict(),
                    './dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            path = os.path.join(get_original_cwd(), reconstructor.cfg.finetuned_params_path 
                if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
            reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
            with torch.no_grad():
                reconstructor.model.eval()
                recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
            recon = recon[0, 0].cpu().numpy()
        else:
            raise NotImplementedError
        
        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

        cfg.mrglik.optim.scl_fct_gamma = observation.shape[-1] * observation.shape[-2]
        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

        recon = torch.from_numpy(recon[None, None])
        if cfg.linearize_weights:
            linearized_weights, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
        else:
            linearized_weights = None
            lin_pred = None

        # type-II MAP
        modules = bayesianized_model.get_all_modules_under_prior()
        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, use_copy='share_parameters')
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)

        proj_recon = ray_trafos['ray_trafo_module'](recon.to(bayesianized_model.store_device))

        log_noise_model_variance_obs = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (recon, proj_recon),
            ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, 
            linearized_weights=linearized_weights, 
            comment = '_recon_num_' + str(i)
            )

        torch.save(bayesianized_model.state_dict(), 
            './bayesianized_model_{}.pt'.format(i))
        torch.save({'log_noise_model_variance_obs': log_noise_model_variance_obs},
            './log_noise_model_variance_obs_{}.pt'.format(i))

        approx_log_prob, block_masks, block_log_probs, block_diags = predictive_image_log_prob(
                recon.to(reconstructor.device), example_image.to(reconstructor.device),
                ray_trafos, bayesianized_model, filtbackproj.to(reconstructor.device), reconstructor.model,
                fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                eps=1e-6, cov_image_eps=1e-6,
                block_size=cfg.density.block_size_for_approx_log_det,
                vec_batch_size=cfg.mrglik.impl.vec_batch_size)

        torch.save({'approx_log_prob': approx_log_prob, 'block_masks': block_masks, 'block_log_probs': block_log_probs, 'block_diags': block_diags},
            './predictive_image_log_prob_{}.pt'.format(i))

        print('approx log prob ', approx_log_prob / example_image.numel())


if __name__ == '__main__':
    coordinator()
