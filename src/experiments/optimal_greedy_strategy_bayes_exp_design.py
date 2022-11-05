from itertools import islice
import numpy as np
import hydra
import odl
from math import ceil
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset, 
        load_testset_rectangles_dataset
        )
from dataset.mnist import simulate
import torch
from deep_image_prior.utils import PSNR, SSIM
from scalable_bayes_exp_design import greedy_optimal_angle_search, adjust_filtbackproj_module
from odl.contrib.torch import OperatorModule

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)
    
    ray_trafos_full = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True, override_angular_sub_sampling=1)
    adjusted_filtbackproj_module = adjust_filtbackproj_module(
        ray_trafos_full['space'], 
        ray_trafos_full['geometry']
    )
    override_angular_sub_sampling = cfg.beam_num_angle // ( cfg.bed.total_num_acq_projs + ceil(cfg.beam_num_angle / cfg.angular_sub_sampling) )
    eqdist_ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True, override_angular_sub_sampling=override_angular_sub_sampling)

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    elif cfg.name == 'rectangles':
        loader = load_testset_rectangles_dataset(cfg)
    elif cfg.name == 'walnut':
        raise NotImplementedError
    else:
        raise NotImplementedError

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
            if cfg.use_double:
                ray_trafos_full['ray_trafo_module'].to(torch.float64)
                ray_trafos_full['ray_trafo_module_adj'].to(torch.float64)
                eqdist_ray_trafos['ray_trafo_module'].to(torch.float64)
                eqdist_ray_trafos['ray_trafo_module_adj'].to(torch.float64)

            observation_full, _, example_image = simulate(
                example_image.double() if cfg.use_double else example_image, 
                ray_trafos_full,
                cfg.noise_specs
                )
                
            if cfg.use_double:
                observation_full = observation_full.double()
                example_image = example_image.double()

            eqdist_observation = observation_full[:, :, ::override_angular_sub_sampling, :]
            eqdist_filtbackproj = adjusted_filtbackproj_module(
                eqdist_observation, 
                eqdist_ray_trafos['ray_trafo_module_adj']
            )
            
            assert torch.allclose(  adjusted_filtbackproj_module(
                eqdist_observation, 
                OperatorModule(odl.tomo.RayTransform(eqdist_ray_trafos['space'], eqdist_ray_trafos['geometry']).adjoint)
                ), eqdist_ray_trafos['pseudoinverse_module'](eqdist_observation), atol=1e-7
            )
            print('equidistance reconstruction of sample {:d}'.format(i))
            print('PSNR:', PSNR(eqdist_filtbackproj[0, 0].numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(eqdist_filtbackproj[0, 0].numpy(), example_image[0, 0].cpu().numpy()))
            
        elif cfg.name == 'walnut':
            raise NotImplementedError
        else:
            raise NotImplementedError

        if cfg.name in ['mnist', 'kmnist', 'rectangles']:

            full_angles = ray_trafos_full['geometry'].angles
            full_num_angles = len(full_angles)
            init_angle_inds = np.arange(0, full_num_angles, cfg.angular_sub_sampling)
            acq_angle_inds = np.setdiff1d(np.arange(full_num_angles), np.arange(0, full_num_angles, cfg.angular_sub_sampling))

            proj_inds_per_angle = np.arange(np.prod(ray_trafos_full['ray_trafo'].range.shape)).reshape(ray_trafos_full['ray_trafo'].range.shape)
            assert proj_inds_per_angle.shape[0] == full_num_angles
            init_proj_inds_list = proj_inds_per_angle[init_angle_inds]
        
            ray_trafo_full_mat_flat = ray_trafos_full['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos_full['space'].shape))
            greedy_optimal_angle_search( 
                    example_image, observation_full, ray_trafos_full, ray_trafo_full_mat_flat,
                    cfg.bed.total_num_acq_projs,
                    proj_inds_per_angle, init_proj_inds_list,
                    acq_angle_inds, init_angle_inds, 
                    log_path=cfg.bed.log_path, 
                    eqdist_filtbackproj=eqdist_filtbackproj
                )
        elif cfg.name == 'walnut':
            raise NotImplementedError  # TODO from walnut_interface
        else:
            raise NotImplementedError


if __name__ == '__main__':
    coordinator()
