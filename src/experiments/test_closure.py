import os
from itertools import islice
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
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, prior_cov_obs_mat_mul,
        get_unet_batch_ensemble, get_fwAD_model)

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True)

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
        v = observation.repeat(cfg.mrglik.impl.vec_batch_size, 1, 1, 1).to(reconstructor.device)
        log_noise_model_variance_obs = torch.tensor(0.).to(reconstructor.device)
        compare_with_assembled_jac_mat_mul = cfg.name in ['mnist', 'kmnist']

        if compare_with_assembled_jac_mat_mul:
            jac = compute_jacobian_single_batch(
                    filtbackproj.to(reconstructor.device),
                    reconstructor.model,
                    modules, example_image.numel())
            from priors_marglik import BlocksGPpriors
            block_priors = BlocksGPpriors(
                    reconstructor.model,
                    bayesianized_model,
                    reconstructor.device,
                    cfg.mrglik.priors.lengthscale_init,
                    cfg.mrglik.priors.variance_init,
                    lin_weights=None)
            v_image_assembled_jac = ray_trafos['ray_trafo_module_adj'](v).view(v.shape[0], -1)

            Kyy = block_priors.matrix_prior_cov_mul(jac) @ jac.transpose(1, 0)
            v_image_assembled_jac = v_image_assembled_jac @ Kyy
            # alternative implementation:
            # v_params_assembled_jac = v_image_assembled_jac @ jac
            # print('v_params_assembled_jac', v_params_assembled_jac[:, :10])
            # v_params_assembled_jac = block_priors.matrix_prior_cov_mul(v_params_assembled_jac)
            # # v_params_assembled_jac = vec_weight_prior_cov_mul(bayesianized_model, v_params_assembled_jac)
            # v_image_assembled_jac = v_params_assembled_jac @ jac.T

            v_obs_assembled_jac = ray_trafos['ray_trafo_module'](v_image_assembled_jac.view(v.shape[0], *example_image.shape[1:]))
            v_obs_assembled_jac = v_obs_assembled_jac + v * torch.exp(log_noise_model_variance_obs)

        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, v.shape[0], return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, use_copy='share_parameters')
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        if cfg.mrglik.impl.use_fwAD_for_jvp:
            print('using forward-mode AD')
            v_obs = prior_cov_obs_mat_mul(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, v, log_noise_model_variance_obs, use_fwAD_for_jvp=True)
        else:
            print('using finite differences')
            v_obs = prior_cov_obs_mat_mul(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, be_model, be_modules, v, log_noise_model_variance_obs, use_fwAD_for_jvp=False)

        if compare_with_assembled_jac_mat_mul:
            print('asserting result is close to the one with assembled jacobian matrix:')
            assert torch.allclose(v_obs, v_obs_assembled_jac)
            print('passed')
        else:
            print('did not assemble jacobian matrix')

    print('max GPU memory used:', torch.cuda.max_memory_allocated())

if __name__ == '__main__':
    coordinator()
