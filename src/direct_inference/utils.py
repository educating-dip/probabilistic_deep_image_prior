import numpy as np 
import jax.numpy as jnp
import PIL
from PIL import Image, ImageOps

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from odl.contrib.torch import OperatorModule
import odl
import torch




def load_jpeg(img_path, size=(64, 64)):

    # Open the image form working directory
    image = Image.open(img_path)
    image = ImageOps.grayscale(image)

    # summarize some details about the image
    print('original size', image.size)
    print(image.mode)

    # Resize image
    resize_im = image.resize(size, resample=PIL.Image.BICUBIC) #PIL.Image.NEAREST

    print('new size', resize_im.size)
    # show the image
    return np.array(resize_im)


def load_KMNIST_dataset(kmnist_path, batchsize=1, train=False):
    
    # We dont apply data standarisation because that way the data is naturally scalled in 0,1
    # transforms.Normalize((0.1307,), (0.3081,))
    # Also, we know the trainset mean is 0.1307
    testset = datasets.KMNIST(root=kmnist_path, train=train, download=True,
                             transform=transforms.Compose([
                                transforms.ToTensor(),
                                ]))
    return DataLoader(testset, batchsize, shuffle=False)


def normalise_img(x):
    return (x - x.min()) / (x.max() - x.min())

def psnr(x, y, smax=1):
    assert x.max() <=  smax
    assert y.max() <=  smax
    assert x.shape == y.shape
    
    mse = ((x-y) ** 2).mean()
    psnr = 10 * np.log10( (smax ** 2) / mse )
    return psnr

def TV(x):
    h_tv = jnp.abs(jnp.diff(x, axis=-1, n=1)).sum()
    v_tv = jnp.abs(jnp.diff(x, axis=-2, n=1)).sum()
    return h_tv + v_tv

def generate_dist_mtx(side):
    coords = np.stack([np.repeat(np.arange(side), side), np.tile(np.arange(side), side)], axis=1)
    coords_exp1 = coords[:,None,:]
    coords_exp0 = coords[None,:,:]
    dist_mtx = ((coords_exp1 - coords_exp0) ** 2).sum(axis=-1) ** 0.5
    return dist_mtx

        
def RadialBasisFuncCov(side, marg_var, AR_p):
    eps = 1e-5        
    AR_p = jnp.clip(AR_p, a_min=eps, a_max=1-eps)
    log_ar_p = jnp.log(AR_p) # lengthscale = -1 / log_ar_p
    dist_mtx = generate_dist_mtx(side)
    cov_mat = marg_var * (jnp.exp(dist_mtx * log_ar_p) + eps * jnp.eye(side ** 2))
    return  cov_mat

def expected_TV(sidelength, marg_var, AR_p):
    return (4 * (sidelength - 1) * sidelength / np.sqrt(np.pi)) * ( (marg_var - marg_var*AR_p ) ** 0.5) 


def N_TV_entries(side):
    return 2 * (side - 1) * side

#######

def get_standard_ray_trafos(size=28, num_angles=5, return_torch_module=True):

    half_size = size / 2
    space = odl.uniform_discr([-half_size, -half_size], [half_size,
                              half_size], [size, size],
                              dtype='float32')
    geometry = odl.tomo.parallel_beam_geometry(space,
            num_angles=num_angles)
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

def gen_op_mat(img_side, num_angles):
    ray_trafos = get_standard_ray_trafos(size=img_side, num_angles=num_angles, return_torch_module=False)
    op_tensor = odl.operator.oputils.matrix_representation(ray_trafos['ray_trafo'])
    op_mat = np.reshape(op_tensor, (-1, img_side**2))
    return op_mat


def simulate(x, ray_trafos, noise_std):

    obs = ray_trafos['ray_trafo'](x)  # this is the transform
    relative_stddev = torch.mean(torch.abs(obs))
    observation = obs + torch.zeros(*obs.shape).normal_(0, 1) \
        * relative_stddev * noise_std
    filtbackproj = ray_trafos['pseudoinverse'](observation)
    return (observation, filtbackproj, x)