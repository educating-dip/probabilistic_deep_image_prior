import os
from itertools import islice
import numpy as np
import random
import hydra
import warnings
from gpytorch.utils.warnings import NumericalWarning
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_rectangles_dataset, load_testset_walnut_patches_dataset, 
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import scipy
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from linearized_weights import weights_linearization
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        optim_marginal_lik_low_rank)

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    if cfg.ignore_gpytorch_numerical_warnings:
        warnings.simplefilter('ignore', NumericalWarning)

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
    elif cfg.name == 'rectangles':
        loader = load_testset_rectangles_dataset(cfg)
    elif cfg.name == 'walnut_patches':
        loader = load_testset_walnut_patches_dataset(cfg)
    else:
        raise NotImplementedError

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
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
            torch.save({'observation': observation, 'filtbackproj': filtbackproj, 'ground_truth': example_image},
                    './sample_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist', 'rectangles', 'walnut_patches']:
            if cfg.load_dip_models_from_path is not None:
                path = os.path.join(cfg.load_dip_models_from_path, 'dip_model_{}.pt'.format(i))
                print('loading model for {} reconstruction from {}'.format(cfg.name, path))
                reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
                with torch.no_grad():
                    reconstructor.model.eval()
                    recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
                recon = recon[0, 0].cpu().numpy()
            else:
                recon, _ = reconstructor.reconstruct(
                    observation, fbp=filtbackproj, ground_truth=example_image)
                torch.save(reconstructor.model.state_dict(),
                        './dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            if cfg.load_dip_models_from_path is not None:
                raise NotImplementedError('model for walnut reconstruction cannot be loaded from a previous run, use net.finetuned_params_path instead')
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

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

        recon = torch.from_numpy(recon[None, None])
        if cfg.linearize_weights:
            linearized_weights, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))

            torch.save({'linearized_weights': linearized_weights, 'linearized_prediction': lin_pred},  
                './linearized_weights_{}.pt'.format(i))

        else:
            linearized_weights = None
            lin_pred = None
        
        # type-II MAP
        modules = bayesianized_model.get_all_modules_under_prior()
        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(bayesianized_model.store_device)
        ray_trafos['ray_trafo_module_adj'].to(bayesianized_model.store_device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)

        proj_recon = ray_trafos['ray_trafo_module'](recon.to(bayesianized_model.store_device))

        log_noise_model_variance_obs = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (recon, proj_recon),
            ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, 
            linearized_weights=linearized_weights, 
            comment = '_recon_num_' + str(i),
            )

        torch.save(bayesianized_model.state_dict(), 
            './bayesianized_model_{}.pt'.format(i))
        torch.save({'log_noise_model_variance_obs': log_noise_model_variance_obs},
            './log_noise_model_variance_obs_{}.pt'.format(i))


if __name__ == '__main__':
    coordinator()
