import os
import numpy as np
from itertools import islice
import hydra
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
import torch.autograd.forward_ad as fwAD
from tqdm import tqdm
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from linearized_laplace import compute_jacobian_single_batch, agregate_flatten_weight_grad
from linearized_weights.linearized_weights import finite_diff_JvP
from scalable_linearised_laplace.batch_ensemble_unet import get_unet_batch_ensemble
from scalable_linearised_laplace.batch_ensemble import Conv2dBatchEnsemble
from scalable_linearised_laplace.fwAD import get_fwAD_model
from linearized_weights.conv2d_fwAD import Conv2dFwAD
from priors_marglik import BayesianizeModel

# note: after calling fwAD_JvP the weights are no longer
# registered as nn.Parameters of the module; to revert this, call
# revert_all_weights_block_to_parameters(modules) afterwards
def fwAD_JvP(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 1
    assert len(x.shape) == 4

    model.eval()

    with fwAD.dual_level():
        set_all_weights_block_tangents(modules, vec)
        out = model(x)[0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

# note: after calling fwAD_JvP_batch_ensemble the weights are no longer
# registered as nn.Parameters of the module; to revert this, call
# revert_all_weights_block_to_parameters(modules) afterwards
def fwAD_JvP_batch_ensemble(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()

    with fwAD.dual_level():
        set_all_weights_block_tangents_batch_ensemble(modules, vec)
        out = model(x)[0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

def set_all_weights_block_tangents(modules, tangents):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dFwAD)
        n_weights = layer.weight.numel()
        weight = layer.weight
        delattr(layer, 'weight')
        setattr(layer, 'weight', fwAD.make_dual(weight, tangents[n_weights_all:n_weights_all+n_weights].view_as(weight)))
        n_weights_all += n_weights

def set_all_weights_block_tangents_batch_ensemble(modules, tangents):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        weight = layer.weight
        delattr(layer, 'weight')
        setattr(layer, 'weight', fwAD.make_dual(weight, tangents[:, n_weights_all:n_weights_all+n_weights].view_as(weight)))
        n_weights_all += n_weights

def revert_all_weights_block_to_parameters(modules):
    for layer in modules:
        assert isinstance(layer, (Conv2dFwAD, Conv2dBatchEnsemble))
        layer.weight = torch.nn.Parameter(fwAD.unpack_dual(layer.weight).primal)

def finite_diff_JvP_batch_ensemble(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec_batch_ensemble(modules)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block_batch_ensemble(modules, w_plus)
        f_plus = model(x)[0]

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block_batch_ensemble(modules, w_minus)
        f_minus = model(x)[0]

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block_batch_ensemble(modules, map_weights)
        return JvP

def set_all_weights_block_batch_ensemble(modules, weights):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        layer.weight.copy_(weights[:, n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
        n_weights_all += n_weights

def get_weight_block_vec_batch_ensemble(modules):
    ws = []
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        ws.append(layer.weight.view(layer.num_instances, -1))
    return torch.cat(ws, dim=1)


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    torch.manual_seed(0)

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True)

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space'],
                 }
    out_dim = np.prod(ray_trafo['reco_space'].shape)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)
    model = reconstructor.model
    model.eval()
    bayesianize_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)
    modules = bayesianize_model.get_all_modules_under_prior()

    num_instances = 10
    be_model, be_module_mapping = get_unet_batch_ensemble(model, num_instances, return_module_mapping=True)
    be_modules = [be_module_mapping[m] for m in modules]

    fwAD_model, fwAD_module_mapping = get_fwAD_model(model, return_module_mapping=True)
    fwAD_modules = [fwAD_module_mapping[m] for m in modules]

    fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True)
    fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

    x_input = torch.rand((1, 1,) + ray_trafo['reco_space'].shape, device=reconstructor.device)
    jac = compute_jacobian_single_batch(x_input, model, modules, out_dim)

    w_tests = torch.rand((2500, jac.shape[1]), device=reconstructor.device)

    x_mat_mul_tests = w_tests @ jac.T

    x_tests = []

    for i in tqdm(range(w_tests.shape[0])):

        x = finite_diff_JvP(x_input, reconstructor.model, w_tests[i], modules).detach().flatten()
        x_tests.append(x)

    for i in range(min(10, w_tests.shape[0])):
        print('max  |x_fd-x|', torch.max(torch.abs(x_tests[i] - x_mat_mul_tests[i])))
        print('mean |x_fd-x|', torch.mean(torch.abs(x_tests[i] - x_mat_mul_tests[i])))
        print('mean |x|     ', torch.mean(torch.abs(x_mat_mul_tests[i])))

    x_fwAD_tests = []

    for i in tqdm(range(w_tests.shape[0])):

        x = fwAD_JvP(x_input, fwAD_model, w_tests[i], fwAD_modules).detach().flatten()
        x_fwAD_tests.append(x)

    for i in range(min(10, w_tests.shape[0])):
        print('max  |x_fwAD-x|', torch.max(torch.abs(x_fwAD_tests[i] - x_mat_mul_tests[i])))
        print('mean |x_fwAD-x|', torch.mean(torch.abs(x_fwAD_tests[i] - x_mat_mul_tests[i])))
        print('mean |x|       ', torch.mean(torch.abs(x_mat_mul_tests[i])))

    x_be_tests = []
    for idx in tqdm(range(0, w_tests.shape[0], num_instances)):
        w_tests_batch = w_tests[idx:idx+num_instances]
        pad_instances = 0
        if w_tests_batch.shape[0] < num_instances:
            pad_instances = num_instances - w_tests_batch.shape[0]
            w_tests_batch = torch.cat([
                    w_tests_batch,
                    torch.zeros(
                            num_instances - w_tests_batch.shape[0], *w_tests_batch.shape[1:],
                            dtype=w_tests_batch.dtype, device=w_tests_batch.device)])
        # print(w_tests_batch.shape)
        x_be_tests_batch = finite_diff_JvP_batch_ensemble(x_input, be_model, w_tests_batch, be_modules).detach().view(num_instances, -1)
        # print(x_be_tests_batch.shape)
        if pad_instances > 0:
            x_be_tests_batch = x_be_tests_batch[:-pad_instances]
        x_be_tests.append(x_be_tests_batch)
    x_be_tests = torch.cat(x_be_tests)

    for i in range(min(10, w_tests.shape[0])):

        print('max  |x_fd_be-x|', torch.max(torch.abs(x_be_tests[i] - x_mat_mul_tests[i])))
        print('mean |x_fd_be-x|', torch.mean(torch.abs(x_be_tests[i] - x_mat_mul_tests[i])))
        print('mean |x|        ', torch.mean(torch.abs(x_mat_mul_tests[i])))

        # import matplotlib.pyplot as plt
        # plt.subplot(1, 3, 1)
        # plt.imshow(x_be_tests[i].view(*ray_trafo['reco_space'].shape).detach().cpu().numpy())
        # plt.colorbar()
        # plt.subplot(1, 3, 2)
        # plt.imshow(x_mat_mul_tests[i].view(*ray_trafo['reco_space'].shape).detach().cpu().numpy())
        # plt.colorbar()
        # plt.subplot(1, 3, 3)
        # plt.imshow((x_be_tests[i]-x_mat_mul_tests[i]).view(*ray_trafo['reco_space'].shape).detach().cpu().numpy())
        # plt.colorbar()
        # plt.show()

    x_fwAD_be_tests = []
    for idx in tqdm(range(0, w_tests.shape[0], num_instances)):
        w_tests_batch = w_tests[idx:idx+num_instances]
        pad_instances = 0
        if w_tests_batch.shape[0] < num_instances:
            pad_instances = num_instances - w_tests_batch.shape[0]
            w_tests_batch = torch.cat([
                    w_tests_batch,
                    torch.zeros(
                            num_instances - w_tests_batch.shape[0], *w_tests_batch.shape[1:],
                            dtype=w_tests_batch.dtype, device=w_tests_batch.device)])
        # print(w_tests_batch.shape)
        x_fwAD_be_tests_batch = fwAD_JvP_batch_ensemble(x_input, fwAD_be_model, w_tests_batch, fwAD_be_modules).detach().view(num_instances, -1)
        # print(x_be_tests_batch.shape)
        if pad_instances > 0:
            x_fwAD_be_tests_batch = x_fwAD_be_tests_batch[:-pad_instances]
        x_fwAD_be_tests.append(x_fwAD_be_tests_batch)
    x_fwAD_be_tests = torch.cat(x_fwAD_be_tests)

    revert_all_weights_block_to_parameters(fwAD_be_modules)
    # print(list(x for x, _ in fwAD_be_model.named_parameters()))

    for i in range(min(10, w_tests.shape[0])):

        print('max  |x_fwAD_be-x|', torch.max(torch.abs(x_fwAD_be_tests[i] - x_mat_mul_tests[i])))
        print('mean |x_fwAD_be-x|', torch.mean(torch.abs(x_fwAD_be_tests[i] - x_mat_mul_tests[i])))
        print('mean |x|          ', torch.mean(torch.abs(x_mat_mul_tests[i])))


if __name__ == '__main__':
    coordinator()
