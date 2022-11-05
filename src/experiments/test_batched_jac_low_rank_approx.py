import os
import time
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
import odl
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks,  get_unet_batch_ensemble, get_fwAD_model, get_batched_jac_low_rank)

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

    torch.manual_seed(0)
    np.random.seed(0)

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            example_image, _ = data_sample
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

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)
        modules = bayesianized_model.get_all_modules_under_prior()
        v = torch.rand(cfg.mrglik.impl.vec_batch_size, 1, *ray_trafos['ray_trafo'].range.shape).to(reconstructor.device)
        compare_with_assembled_jac_mat_mul = cfg.name in ['mnist', 'kmnist']

        if compare_with_assembled_jac_mat_mul:
            jac = compute_jacobian_single_batch(
                    filtbackproj.to(reconstructor.device),
                    reconstructor.model,
                    modules, example_image.numel())

        add_batch_grad_hooks(reconstructor.model, modules)

        reduced_rank_dim = cfg.density.low_rank_jacobian.low_rank_dim + cfg.density.low_rank_jacobian.oversampling_param
        random_matrix = torch.randn(
            (bayesianized_model.num_params_under_priors,
                    reduced_rank_dim, 
                ),
            device=bayesianized_model.store_device
            )
        start = time.perf_counter()

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, v.shape[0], return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True)
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        U, S, Vh = get_batched_jac_low_rank(random_matrix, filtbackproj.to(reconstructor.device),
            bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules,
            cfg.mrglik.impl.vec_batch_size, cfg.density.low_rank_jacobian.low_rank_dim,
             cfg.density.low_rank_jacobian.oversampling_param,
            use_cpu=cfg.density.low_rank_jacobian.use_cpu)
        print('time in sec to approx cov_obs_mat {:.4f}'.format(time.perf_counter() - start))

        low_rank_jac_mat = U @ torch.diag_embed(S) @ Vh
                
        im_domain = odl.rn([filtbackproj.numel()], dtype=np.float32)
        parms_domain = odl.rn([bayesianized_model.num_params_under_priors], dtype=np.float32)

        low_rank_jac_op = odl.operator.tensor_ops.MatrixOperator(low_rank_jac_mat.cpu().numpy(),
            domain=parms_domain, range=im_domain)
        
        jac_op = odl.operator.tensor_ops.MatrixOperator(jac.cpu().numpy(),
            domain=parms_domain, range=im_domain)

        print('opnorms')
        print('opnorm of jac_op', odl.operator.oputils.power_method_opnorm(jac_op, maxiter=100, rtol=1e-05, atol=1e-08))
        print('opnorm of low_rank_jac_op', odl.operator.oputils.power_method_opnorm(low_rank_jac_op, maxiter=100, rtol=1e-05, atol=1e-08))

        print('differences')
        print('opnorm of low_rank_jac_op-jac_op', odl.operator.oputils.power_method_opnorm(low_rank_jac_op-jac_op, maxiter=100, rtol=1e-05, atol=1e-08))

    print('max GPU memory used:', torch.cuda.max_memory_allocated())

if __name__ == '__main__':
    coordinator()
