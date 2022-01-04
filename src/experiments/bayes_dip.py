import os
from itertools import islice
import numpy as np
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
from priors_marglik import BayesianizeModel, BlocksGPpriors
from linearized_weights import weights_linearization
from linearized_laplace import compute_jacobian_single_batch
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        optim_marginal_lik_low_rank,
        get_prior_cov_obs_mat, get_predictive_cov_image_block,
        get_exact_predictive_cov_image_mat)
from linearized_laplace import image_space_lin_model_post_pred_cov
from scalable_linearised_laplace.density import get_cov_image_mat

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

        cfg.mrglik.optim.scl_fct_gamma = observation.shape[-1] * observation.shape[-2]
        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)

        recon = torch.from_numpy(recon[None, None])
        if cfg.linearize_weights:
            optim_lin_params, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
        else:
            optim_lin_params = None
            lin_pred = None

        jac = compute_jacobian_single_batch(
                filtbackproj.to(reconstructor.device),
                reconstructor.model,
                bayesianized_model.get_all_modules_under_prior(), example_image.numel())

        # type-II MAP
        modules = bayesianized_model.get_all_modules_under_prior()
        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, cfg.mrglik.impl.vec_batch_size, return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, use_copy='share_parameters')
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        proj_recon = ray_trafos['ray_trafo_module'](recon.to(bayesianized_model.store_device))

        log_noise_model_variance_obs = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (recon, proj_recon),
            ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules,
            comment = '_recon_num_' + str(i)
            )

        torch.save(bayesianized_model.state_dict(), 
            './bayesianized_model_{}.pt'.format(i))

        block_priors = BlocksGPpriors(
            reconstructor.model,
            bayesianized_model,
            reconstructor.device,
            cfg.mrglik.priors.lengthscale_init,
            cfg.mrglik.priors.variance_init,
            lin_weights=None
            )

        predictive_cov_image_exact = get_exact_predictive_cov_image_mat(ray_trafos, bayesianized_model, jac, log_noise_model_variance_obs, eps=1e-6, cov_image_eps=1e-6)
        print('exact diag min, max: ', torch.min(predictive_cov_image_exact.diag()), torch.mean(predictive_cov_image_exact.diag()), torch.max(predictive_cov_image_exact.diag()))
        print('exact diag min**0.5, max**0.5: ', torch.min(predictive_cov_image_exact.diag())**0.5, torch.mean(predictive_cov_image_exact.diag())**0.5, torch.max(predictive_cov_image_exact.diag())**0.5)

        ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
        ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(reconstructor.device)
        jac_obs = ray_trafo_mat.cuda() @ jac
        _, _, Kxx_alternative = image_space_lin_model_post_pred_cov(block_priors, jac, jac_obs, torch.exp(log_noise_model_variance_obs))
        Kyy_alternative = block_priors.matrix_prior_cov_mul(jac_obs) @ jac_obs.transpose(1, 0)
        Kyy_alternative[np.diag_indices(Kyy_alternative.shape[0])] += torch.exp(log_noise_model_variance_obs)

        predictive_cov_image_exact_Kyy_alternative = get_exact_predictive_cov_image_mat(ray_trafos, bayesianized_model, jac, log_noise_model_variance_obs, eps=1e-6, cov_image_eps=1e-6, cov_obs_mat=Kyy_alternative)
        print('alternative Kyy exact diag min, max: ', torch.min(predictive_cov_image_exact_Kyy_alternative.diag()), torch.mean(predictive_cov_image_exact_Kyy_alternative.diag()), torch.max(predictive_cov_image_exact_Kyy_alternative.diag()))
        print('alternative Kyy exact diag min**0.5, max**0.5: ', torch.min(predictive_cov_image_exact_Kyy_alternative.diag())**0.5, torch.mean(predictive_cov_image_exact_Kyy_alternative.diag())**0.5, torch.max(predictive_cov_image_exact_Kyy_alternative.diag())**0.5)

        ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
        ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(reconstructor.device)
        jac_y = ray_trafo_mat.cuda() @ jac
        _, predictive_cov_image_exact2, Kff2 = image_space_lin_model_post_pred_cov(block_priors, jac, jac_y, torch.exp(log_noise_model_variance_obs))
        print('exact2 diag min, max: ', torch.min(predictive_cov_image_exact2.diag()), torch.mean(predictive_cov_image_exact2.diag()), torch.max(predictive_cov_image_exact2.diag()))
        print('exact2 diag min**0.5, max**0.5: ', torch.min(predictive_cov_image_exact2.diag())**0.5, torch.mean(predictive_cov_image_exact2.diag())**0.5, torch.max(predictive_cov_image_exact2.diag())**0.5)
        Kff = get_cov_image_mat(bayesianized_model, jac, eps=1e-6)

        mask = np.ones(example_image.numel(), dtype=bool)
        cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size=cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=True)
        predictive_cov_image_block = get_predictive_cov_image_block(mask, cov_obs_mat, ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, vec_batch_size=cfg.mrglik.impl.vec_batch_size, obs_shape=observation.shape[-2:], eps=1e-6, cov_image_eps=1e-6, return_cholesky=False)
        print(torch.min(predictive_cov_image_block.diag()), torch.mean(predictive_cov_image_block.diag()), torch.max(predictive_cov_image_block.diag()))
        print(torch.min(predictive_cov_image_block.diag())**0.5, torch.mean(predictive_cov_image_block.diag())**0.5, torch.max(predictive_cov_image_block.diag())**0.5)
        err = torch.abs(recon - example_image)
        print(torch.min(err), torch.mean(err), torch.max(err))
        breakpoint()
        import matplotlib.pyplot as plt
        plt.imshow(predictive_cov_image_block.diag().view(example_image.shape)[0, 0].detach().cpu().numpy())
        plt.show()

if __name__ == '__main__':
    coordinator()
