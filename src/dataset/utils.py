import os
import odl
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from odl.contrib.torch import OperatorModule
from torch.utils.data import DataLoader
from hydra.utils import get_original_cwd

def load_testset_MNIST_dataset(path='mnist', batchsize=1,
                               crop=False):
    path = os.path.join(get_original_cwd(), path)
    testset = datasets.MNIST(root=path, train=False, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                (0.1307,), (0.3081,))]))
    return DataLoader(testset, batchsize, shuffle=False)

def get_standard_ray_trafos(cfg, return_torch_module=True):

    half_size = cfg.size / 2
    space = odl.uniform_discr([-half_size, -half_size], [half_size,
                              half_size], [cfg.size, cfg.size],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space,
            num_angles=cfg.beam_num_angle)
    ray_trafo = odl.tomo.RayTransform(space, geometry)
    pseudoinverse = odl.tomo.fbp_op(ray_trafo)
    if return_torch_module:
        ray_trafo = OperatorModule(ray_trafo)
        pseudoinverse = OperatorModule(pseudoinverse)

    return {
        'space': space,
        'geometry': geometry,
        'ray_trafo': ray_trafo,
        'pseudoinverse': pseudoinverse,
        }
