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
from scalable_linearised_laplace.batch_ensemble_unet import get_unet_batch_ensemble
from scalable_linearised_laplace.batch_ensemble import Conv2dBatchEnsemble
from scalable_linearised_laplace.fwAD import get_fwAD_model
from scalable_linearised_laplace.jvp import (
    finite_diff_JvP, finite_diff_JvP_batch_ensemble,
    fwAD_JvP, fwAD_JvP_batch_ensemble, FwAD_JvP_PreserveAndRevertWeightsToParameters,
    get_jac_fwAD, get_jac_fwAD_batch_ensemble)
from scalable_linearised_laplace.conv2d_fwAD import Conv2dFwAD
from scalable_linearised_laplace.reduced_model import get_reduced_model, Inactive, Leaf
from scalable_linearised_laplace.reduced_model_unet import get_inactive_and_leaf_modules_unet
from priors_marglik import BayesianizeModel


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

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
    keep_num_blocks = 2
    modules_reduced = [m for m in modules if m in [
            model.up[-1].skip_conv[0],  # 1x1 skip from inc to up[-1]
            model.up[-1].conv[0], model.up[-1].conv[len(model.up[-1].conv)//2],  # 3x3 conv layers in up[-1]
            model.outc.conv]  # 1x1 outc
            ]

    num_instances = 2
    be_model, be_module_mapping = get_unet_batch_ensemble(model, num_instances, return_module_mapping=True)
    be_modules = [be_module_mapping[m] for m in modules]
    be_modules_reduced = [be_module_mapping[m] for m in modules_reduced]

    fwAD_model, fwAD_module_mapping = get_fwAD_model(model, return_module_mapping=True, share_parameters=True)
    fwAD_modules = [fwAD_module_mapping[m] for m in modules]
    fwAD_modules_reduced = [fwAD_module_mapping[m] for m in modules_reduced]

    fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, share_parameters=True)
    fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]
    fwAD_be_modules_reduced = [fwAD_be_module_mapping[m] for m in be_modules_reduced]

    assert cfg.name in ['mnist', 'kmnist']

    x_input = torch.rand((1, 1,) + ray_trafo['reco_space'].shape, device=reconstructor.device)

    fwAD_inactive_modules, fwAD_leaf_modules = get_inactive_and_leaf_modules_unet(fwAD_model, keep_num_blocks=keep_num_blocks)
    reduced_fwAD_model, reduced_fwAD_module_mapping = get_reduced_model(
        fwAD_model, x_input,
        replace_inactive=fwAD_inactive_modules, replace_leaf=fwAD_leaf_modules, return_module_mapping=True, share_parameters=True)
    reduced_fwAD_modules_reduced = [reduced_fwAD_module_mapping[m] for m in fwAD_modules_reduced]
    assert all(m in reduced_fwAD_model.modules() for m in reduced_fwAD_modules_reduced), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'

    fwAD_be_inactive_modules, fwAD_be_leaf_modules = get_inactive_and_leaf_modules_unet(fwAD_be_model, keep_num_blocks=keep_num_blocks)
    reduced_fwAD_be_model, reduced_fwAD_be_module_mapping = get_reduced_model(
        fwAD_be_model, torch.broadcast_to(x_input, (num_instances,) + x_input.shape),
        replace_inactive=fwAD_be_inactive_modules, replace_leaf=fwAD_be_leaf_modules, return_module_mapping=True, share_parameters=True)
    reduced_fwAD_be_modules_reduced = [reduced_fwAD_be_module_mapping[m] for m in fwAD_be_modules_reduced]
    assert all(m in reduced_fwAD_be_model.modules() for m in reduced_fwAD_be_modules_reduced), 'some module(s) in the reduced set of modules under prior cannot be found in the reduced unet model; usually this indicates that get_inactive_and_leaf_modules_unet() was called with a too small keep_num_blocks'

    jac = compute_jacobian_single_batch(x_input, model, modules, out_dim)
    # jac_fwAD = get_jac_fwAD(x_input, fwAD_model, fwAD_modules)  # too slow
    jac_reduced = compute_jacobian_single_batch(x_input, model, modules_reduced, out_dim)
    jac_reduced_fwAD = get_jac_fwAD(x_input, fwAD_model, fwAD_modules_reduced)
    jac_reduced_fwAD_be = get_jac_fwAD_batch_ensemble(x_input, fwAD_be_model, fwAD_be_modules_reduced)

    w_tests = torch.rand((2500, jac.shape[1]), device=reconstructor.device)
    w_tests_reduced = torch.rand((2500, jac_reduced_fwAD.shape[1]), device=reconstructor.device)

    x_mat_mul_tests = w_tests @ jac.T
    # x_mat_mul_tests_fwAD = w_tests @ jac_fwAD.T
    x_mat_mul_tests_reduced = w_tests_reduced @ jac_reduced.T
    x_mat_mul_tests_reduced_fwAD = w_tests_reduced @ jac_reduced_fwAD.T
    x_mat_mul_tests_reduced_fwAD_be = w_tests_reduced @ jac_reduced_fwAD_be.T
    x_closure_tests_reduced_fwAD = []
    for i in tqdm(range(w_tests_reduced.shape[0])):
        x = fwAD_JvP(x_input, fwAD_model, w_tests_reduced[i], fwAD_modules_reduced).detach().flatten()
        x_closure_tests_reduced_fwAD.append(x)
    x_closure_tests_reduced_reduced_fwAD = []
    for i in tqdm(range(w_tests_reduced.shape[0])):
        x = fwAD_JvP(x_input, reduced_fwAD_model, w_tests_reduced[i], reduced_fwAD_modules_reduced).detach().flatten()
        x_closure_tests_reduced_reduced_fwAD.append(x)
    x_closure_tests_reduced_fwAD_be = []
    for idx in tqdm(range(0, w_tests_reduced.shape[0], num_instances)):
        w_tests_reduced_batch = w_tests_reduced[idx:idx+num_instances]
        pad_instances = 0
        if w_tests_reduced_batch.shape[0] < num_instances:
            pad_instances = num_instances - w_tests_reduced_batch.shape[0]
            w_tests_reduced_batch = torch.cat([
                    w_tests_reduced_batch,
                    torch.zeros(
                            num_instances - w_tests_reduced_batch.shape[0], *w_tests_reduced_batch.shape[1:],
                            dtype=w_tests_reduced_batch.dtype, device=w_tests_reduced_batch.device)])
        x_closure_tests_reduced_fwAD_be_batch = fwAD_JvP_batch_ensemble(x_input, fwAD_be_model, w_tests_reduced_batch, fwAD_be_modules_reduced).detach().view(num_instances, -1)
        if pad_instances > 0:
            x_closure_tests_reduced_fwAD_be_batch = x_closure_tests_reduced_fwAD_be_batch[:-pad_instances]
        x_closure_tests_reduced_fwAD_be.append(x_closure_tests_reduced_fwAD_be_batch)
    x_closure_tests_reduced_fwAD_be = torch.cat(x_closure_tests_reduced_fwAD_be)
    x_closure_tests_reduced_reduced_fwAD_be = []
    for idx in tqdm(range(0, w_tests_reduced.shape[0], num_instances)):
        w_tests_reduced_batch = w_tests_reduced[idx:idx+num_instances]
        pad_instances = 0
        if w_tests_reduced_batch.shape[0] < num_instances:
            pad_instances = num_instances - w_tests_reduced_batch.shape[0]
            w_tests_reduced_batch = torch.cat([
                    w_tests_reduced_batch,
                    torch.zeros(
                            num_instances - w_tests_reduced_batch.shape[0], *w_tests_reduced_batch.shape[1:],
                            dtype=w_tests_reduced_batch.dtype, device=w_tests_reduced_batch.device)])
        x_closure_tests_reduced_reduced_fwAD_be_batch = fwAD_JvP_batch_ensemble(x_input, reduced_fwAD_be_model, w_tests_reduced_batch, reduced_fwAD_be_modules_reduced).detach().view(num_instances, -1)
        if pad_instances > 0:
            x_closure_tests_reduced_reduced_fwAD_be_batch = x_closure_tests_reduced_reduced_fwAD_be_batch[:-pad_instances]
        x_closure_tests_reduced_reduced_fwAD_be.append(x_closure_tests_reduced_reduced_fwAD_be_batch)
    x_closure_tests_reduced_reduced_fwAD_be = torch.cat(x_closure_tests_reduced_reduced_fwAD_be)

    # for i in range(min(10, w_tests.shape[0])):
    #     print('max  |x_jac_fwAD-x|', torch.max(torch.abs(x_mat_mul_tests_fwAD[i] - x_mat_mul_tests[i])))
    #     print('mean |x_jac_fwAD-x|', torch.mean(torch.abs(x_mat_mul_tests_fwAD[i] - x_mat_mul_tests[i])))
    #     print('mean |x|     ', torch.mean(torch.abs(x_mat_mul_tests[i])))
    for i in range(min(10, w_tests_reduced.shape[0])):
        print('max  |x_reduced_jac_fwAD_be-x_reduced|                ', torch.max(torch.abs(x_mat_mul_tests_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_fwAD_be-x_reduced|                ', torch.mean(torch.abs(x_mat_mul_tests_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced_jac_fwAD-x_reduced|                   ', torch.max(torch.abs(x_mat_mul_tests_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_fwAD-x_reduced|                   ', torch.mean(torch.abs(x_mat_mul_tests_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced_jac_closure_fwAD_be-x_reduced|        ', torch.max(torch.abs(x_closure_tests_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_closure_fwAD_be-x_reduced|        ', torch.mean(torch.abs(x_closure_tests_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced_jac_closure_fwAD-x_reduced|           ', torch.max(torch.abs(x_closure_tests_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_closure_fwAD-x_reduced|           ', torch.mean(torch.abs(x_closure_tests_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced_jac_closure_reduced_fwAD_be-x_reduced|', torch.max(torch.abs(x_closure_tests_reduced_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_closure_reduced_fwAD_be-x_reduced|', torch.mean(torch.abs(x_closure_tests_reduced_reduced_fwAD_be[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced_jac_closure_reduced_fwAD-x_reduced|   ', torch.max(torch.abs(x_closure_tests_reduced_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('mean |x_reduced_jac_closure_reduced_fwAD-x_reduced|   ', torch.mean(torch.abs(x_closure_tests_reduced_reduced_fwAD[i] - x_mat_mul_tests_reduced[i])))
        print('max  |x_reduced-x|                                    ', torch.max(torch.abs(x_mat_mul_tests_reduced[i] - x_mat_mul_tests[i])))
        print('mean |x_reduced-x|                                    ', torch.mean(torch.abs(x_mat_mul_tests_reduced[i] - x_mat_mul_tests[i])))
        print('mean |x|                                              ', torch.mean(torch.abs(x_mat_mul_tests[i])))


if __name__ == '__main__':
    coordinator()
