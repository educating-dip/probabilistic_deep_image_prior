import os
import numpy as np
import odl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from odl.contrib.torch import OperatorModule
from torch.utils.data import DataLoader, TensorDataset
from hydra.utils import get_original_cwd
from .walnut import (
        get_walnut_data, get_walnut_single_slice_matrix_ray_trafos,
        get_walnut_proj_numel)
from .pretraining_ellipses import DiskDistributedEllipsesDataset

def load_testset_MNIST_dataset(path='mnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    testset = datasets.MNIST(root=path, train=False, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def load_testset_KMNIST_dataset(path='kmnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    testset = datasets.KMNIST(root=path, train=False, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def load_testset_walnut(cfg):
    observation, filtbackproj, ground_truth = get_walnut_data(cfg)

    testset = TensorDataset(
            torch.from_numpy(observation[None, None]),
            torch.from_numpy(filtbackproj[None, None]),
            torch.from_numpy(ground_truth[None, None]))

    return DataLoader(testset, 1, shuffle=False)

def get_standard_ray_trafos(cfg, return_torch_module=True,
                            return_op_mat=False):

    ray_trafo_impl = cfg.get('ray_trafo_impl', None)

    if not ray_trafo_impl:  # default, create parallel beam geometry
        half_size = cfg.size / 2
        space = odl.uniform_discr([-half_size, -half_size], [half_size,
                                half_size], [cfg.size, cfg.size],
                                dtype='float32')
        geometry = odl.tomo.parallel_beam_geometry(space,
                num_angles=cfg.beam_num_angle)
        ray_trafo = odl.tomo.RayTransform(space, geometry)
        pseudoinverse = odl.tomo.fbp_op(ray_trafo)
        ray_trafo_dict = {
            'space': space,
            'geometry': geometry,
            'ray_trafo': ray_trafo,
            'pseudoinverse': pseudoinverse,
            }

        if return_torch_module:
            ray_trafo_module = OperatorModule(ray_trafo)
            ray_trafo_dict['ray_trafo_module'] = ray_trafo_module
            ray_trafo_module_adj = OperatorModule(ray_trafo.adjoint)
            ray_trafo_dict['ray_trafo_module_adj'] = ray_trafo_module_adj
            pseudoinverse_module = OperatorModule(pseudoinverse)
            ray_trafo_dict['pseudoinverse_module'] = pseudoinverse_module
        if return_op_mat:
            ray_trafo_mat = \
                odl.operator.oputils.matrix_representation(ray_trafo)
            ray_trafo_dict['ray_trafo_mat'] = ray_trafo_mat
            ray_trafo_mat_adj = ray_trafo_mat.T
            ray_trafo_dict['ray_trafo_mat_adj'] = ray_trafo_mat_adj

    elif ray_trafo_impl == 'custom':
        if cfg.ray_trafo_custom.name == 'walnut_single_slice_matrix':
            ray_trafo_dict = get_walnut_single_slice_matrix_ray_trafos(
                    cfg,
                    return_torch_module=return_torch_module,
                    return_op_mat=return_op_mat)
        else:
            raise ValueError('Unknown custom ray trafo \'{}\''.format(
                    cfg.ray_trafo_custom.name))
    else:
        raise ValueError('Unknown ray trafo implementation \'{}\''.format(
                    ray_trafo_impl))

    return ray_trafo_dict


def extract_trafos_as_matrices(ray_trafos): 

    trafo = torch.from_numpy(ray_trafos['ray_trafo_mat'])
    trafo = trafo.view(-1, ray_trafos['space'].shape[0]**2)
    trafo_adj = trafo.T
    trafo_adj_trafo = trafo_adj @ trafo
    trafo_trafo_adj = trafo @ trafo_adj

    return trafo, trafo_adj, trafo_adj_trafo, trafo_trafo_adj


def get_pretraining_dataset(cfg, return_ray_trafo_torch_module=True,
                            return_ray_trafo_op_mat=False):
    ray_trafos = get_standard_ray_trafos(cfg,
            return_torch_module=return_ray_trafo_torch_module,
            return_op_mat=return_ray_trafo_op_mat)

    ray_trafo = ray_trafos['ray_trafo']
    pseudoinverse = ray_trafos['pseudoinverse']

    cfg_p = cfg.pretraining

    if cfg_p.noise_specs.noise_type == 'white':
        specs_kwargs = {
            'stddev': cfg_p.noise_specs.stddev
        }
    elif cfg_p.noise_specs.noise_type == 'poisson':
        specs_kwargs = {
            'mu_water': cfg_p.noise_specs.mu_water,
            'photons_per_pixel': cfg_p.noise_specs.photons_per_pixel
        }
    else:
        raise NotImplementedError

    if cfg.name == 'walnut':
        dataset_specs = {
            'diameter': cfg_p.disk_diameter,
            'image_size': cfg.image_specs.size,
            'train_len': cfg_p.train_len,
            'validation_len': cfg_p.validation_len,
            'test_len': cfg_p.test_len}
        ellipses_dataset = DiskDistributedEllipsesDataset(**dataset_specs)
        space = ellipses_dataset.space
        proj_numel = get_walnut_proj_numel(cfg)
        proj_space = odl.rn((1, proj_numel), dtype=np.float32)
        dataset = ellipses_dataset.create_pair_dataset(ray_trafo=ray_trafo,
                pinv_ray_trafo=pseudoinverse,
                domain=space, proj_space=proj_space,
                noise_type=cfg_p.noise_specs.noise_type,
                specs_kwargs=specs_kwargs,
                noise_seeds={
                    'train': cfg_p.seed,
                    'validation': cfg_p.seed + 1,
                    'test': cfg_p.seed + 2})
    else:
        raise NotImplementedError

    return dataset, ray_trafos
