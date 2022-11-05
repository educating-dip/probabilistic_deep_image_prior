import os
from itertools import islice
import numpy as np
import random
import hydra
import warnings
from tqdm import tqdm
from gpytorch.utils.warnings import NumericalWarning
from omegaconf import DictConfig
from dataset import tSVDMatrixModule
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
from sample_based_mrglik_hyper_optim import sample_based_EM_hyperparams_optim
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model, get_inactive_and_leaf_modules_unet, get_reduced_model, 
        get_batched_jac_low_rank, 
        )

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

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.sample_based_mrglik.priors.lengthscale_init,
            'variance_init':cfg.sample_based_mrglik.priors.variance_init}, include_normal_priors=cfg.sample_based_mrglik.priors.include_normal_priors,
            exclude_gp_priors_list=cfg.sample_based_mrglik.reduced_model.exclude_gp_priors_list, exclude_normal_priors_list=cfg.sample_based_mrglik.reduced_model.exclude_normal_priors_list)

        recon = torch.from_numpy(recon[None, None])

        # type-II MAP
        model = reconstructor.model
        all_modules_under_prior = bayesianized_model.get_all_modules_under_prior()
        modules = all_modules_under_prior
        add_batch_grad_hooks(model, modules)
        model_batched = None

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.sample_based_mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        if cfg.sample_based_mrglik.impl.reduce_model:
            inactive_modules, leaf_modules = get_inactive_and_leaf_modules_unet(reconstructor.model, keep_modules=modules)
            
            reduced_model, reduced_module_mapping = get_reduced_model(
                reconstructor.model, filtbackproj.to(reconstructor.device),
                replace_inactive=inactive_modules, replace_leaf=leaf_modules, return_module_mapping=True, share_parameters=True)
            modules = [reduced_module_mapping[m] for m in all_modules_under_prior]
            assert all(m in reduced_model.modules() for m in modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            model = reduced_model

            reduced_model_batched, reduced_module_batched_mapping = get_reduced_model(
                reconstructor.model, filtbackproj.to(reconstructor.device).expand(cfg.sample_based_mrglik.impl.vec_batch_size, *filtbackproj.shape[1:]),
                replace_inactive=inactive_modules, replace_leaf=leaf_modules, return_module_mapping=True, share_parameters=True)
            modules_batched = [reduced_module_batched_mapping[m] for m in all_modules_under_prior]
            assert all(m in reduced_model_batched.modules() for m in modules_batched), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            model_batched = reduced_model_batched

            fwAD_be_inactive_modules, fwAD_be_leaf_modules = get_inactive_and_leaf_modules_unet(fwAD_be_model, keep_modules=fwAD_be_modules)
            reduced_fwAD_be_model, reduced_fwAD_be_module_mapping = get_reduced_model(
                fwAD_be_model, torch.broadcast_to(filtbackproj.to(reconstructor.device), (cfg.sample_based_mrglik.impl.vec_batch_size,) + filtbackproj.shape),
                replace_inactive=fwAD_be_inactive_modules, replace_leaf=fwAD_be_leaf_modules, return_module_mapping=True, share_parameters=True)
            fwAD_be_modules = [reduced_fwAD_be_module_mapping[m] for m in fwAD_be_modules]
            assert all(m in reduced_fwAD_be_model.modules() for m in fwAD_be_modules), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'
            fwAD_be_model = reduced_fwAD_be_model
        
        jac = None 
        if cfg.sample_based_mrglik.impl.assemble_jac == 'low_rank':
            if cfg.sample_based_mrglik.low_rank_jacobian.load_path is not None:
                jac_dict = torch.load(os.path.join(cfg.sample_based_mrglik.low_rank_jacobian.load_path, 'low_rank_jac.pt'), map_location=reconstructor.device)
                jac = (jac_dict['U'], jac_dict['S'], jac_dict['Vh'])
            else:
                low_rank_rank_dim = cfg.sample_based_mrglik.low_rank_jacobian.low_rank_dim + cfg.sample_based_mrglik.low_rank_jacobian.oversampling_param
                random_matrix = torch.randn(
                    (bayesianized_model.num_params_under_priors,
                            low_rank_rank_dim, 
                        ),
                    device=bayesianized_model.store_device
                    )

                U, S, Vh = get_batched_jac_low_rank(random_matrix, filtbackproj.to(reconstructor.device),
                    bayesianized_model, model if not cfg.sample_based_mrglik.impl.reduce_model else model_batched, fwAD_be_model, fwAD_be_modules,
                    cfg.sample_based_mrglik.impl.vec_batch_size, cfg.sample_based_mrglik.low_rank_jacobian.low_rank_dim,
                    cfg.sample_based_mrglik.low_rank_jacobian.oversampling_param, 
                    use_cpu=cfg.sample_based_mrglik.low_rank_jacobian.use_cpu,
                    return_on_cpu=cfg.sample_based_mrglik.low_rank_jacobian.store_on_cpu
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
        
        if cfg.sample_based_mrglik.impl.use_san_trafo:
                
            san_ray_trafos = ray_trafos.copy()
            
            trafo = ray_trafos['ray_trafo_module'].matrix
            U_trafo, S_trafo, Vh_trafo = torch.svd_lowrank(trafo, q=50)            
            san_trafo = tSVDMatrixModule(tsvd_matrix=(U_trafo, S_trafo, Vh_trafo), out_shape=observation.shape[2:])
            san_trafo_T = tSVDMatrixModule(tsvd_matrix=(U_trafo, S_trafo, Vh_trafo), out_shape=filtbackproj.shape[2:], adjoint=True)
            
            san_ray_trafos['ray_trafo_module'] = san_trafo
            san_ray_trafos['ray_trafo_module_adj'] = san_trafo_T
        else:
            san_ray_trafos = None

        sample_based_EM_hyperparams_optim(
        cfg, 
        ray_trafos, san_ray_trafos, 
        filtbackproj, observation, example_image, 
        bayesianized_model, model, modules, model_batched, fwAD_be_model, fwAD_be_modules, jac
        )

        torch.save(bayesianized_model.state_dict(), 
                    './bayesianized_model_{}.pt'.format(i)
            )

if __name__ == '__main__':
    coordinator()
