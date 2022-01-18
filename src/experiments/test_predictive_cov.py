import os
import time
import numpy as np
from itertools import islice
import hydra
from omegaconf import DictConfig
from copy import deepcopy
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
from linearized_laplace import compute_jacobian_single_batch, image_space_lin_model_post_pred_cov
from scalable_linearised_laplace import (
        add_batch_grad_hooks, prior_cov_obs_mat_mul, get_prior_cov_obs_mat, get_diag_prior_cov_obs_mat,
        get_unet_batch_ensemble, get_fwAD_model, compute_exact_log_det_grad, compute_approx_log_det_grad,
        get_predictive_cov_image_block, get_exact_predictive_cov_image_mat, predictive_image_log_prob,
        sample_from_posterior, approx_density_from_samples, predictive_image_log_prob_from_samples)

# may require a good network reco and/or high precision computation

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

        recon = torch.from_numpy(recon[None, None])

        bayesianized_model = BayesianizeModel(reconstructor, **{'lengthscale_init': cfg.mrglik.priors.lengthscale_init ,
            'variance_init': cfg.mrglik.priors.variance_init}, include_normal_priors=cfg.mrglik.priors.include_normal_priors)
        modules = bayesianized_model.get_all_modules_under_prior()
        v = observation.repeat(cfg.mrglik.impl.vec_batch_size, 1, 1, 1).to(reconstructor.device)
        log_noise_model_variance_obs = torch.tensor(0.).to(reconstructor.device)

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

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
        v_image_assembled_jac = ray_trafos['ray_trafo_module_adj'](v).view(v.shape[0], -1)  # v * A 

        Kxx = block_priors.matrix_prior_cov_mul(jac) @ jac.transpose(1, 0) # J * Sigma_theta * J.T

        # constructing Kyy
        Kyy = ray_trafos['ray_trafo_module'](Kxx.view(example_image.numel(), *example_image.shape[1:]))
        Kyy = Kyy.view(example_image.numel(), -1).T.view(-1, *example_image.shape[1:])
        Kyy = ray_trafos['ray_trafo_module'](Kyy).view(-1, np.prod(v.shape[2:])) + torch.exp(log_noise_model_variance_obs) * torch.eye(np.prod(v.shape[2:]), device=reconstructor.device)

        add_batch_grad_hooks(reconstructor.model, modules)

        be_model, be_module_mapping = get_unet_batch_ensemble(reconstructor.model, v.shape[0], return_module_mapping=True)
        be_modules = [be_module_mapping[m] for m in modules]

        fwAD_be_model, fwAD_be_module_mapping = get_fwAD_model(be_model, return_module_mapping=True, use_copy='share_parameters')
        fwAD_be_modules = [fwAD_be_module_mapping[m] for m in be_modules]

        if cfg.mrglik.impl.use_fwAD_for_jvp:
            print('using forward-mode AD')
            cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=True)
        else:
            print('using finite differences')
            cov_obs_mat = get_prior_cov_obs_mat(ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, be_model, be_modules, log_noise_model_variance_obs, cfg.mrglik.impl.vec_batch_size, use_fwAD_for_jvp=False)

        print('max Kyy-cov_obs_mat', torch.max(torch.abs(Kyy-cov_obs_mat)))

        mask = np.ones(example_image.numel(), dtype=bool)

        cov_obs_mat_chol = torch.linalg.cholesky(cov_obs_mat)
        predictive_cov_image_block = get_predictive_cov_image_block(mask, cov_obs_mat_chol, ray_trafos, filtbackproj.to(reconstructor.device), bayesianized_model, reconstructor.model, fwAD_be_model, fwAD_be_modules, vec_batch_size=cfg.mrglik.impl.vec_batch_size, eps=1e-6, cov_image_eps=1e-6, return_cholesky=False)
        print('approx diag min, mean, max: ', torch.min(predictive_cov_image_block.diag()).item(), torch.mean(predictive_cov_image_block.diag()).item(), torch.max(predictive_cov_image_block.diag()).item())
        print('approx diag min**0.5, mean**0.5, max**0.5: ', torch.min(predictive_cov_image_block.diag()).item()**0.5, torch.mean(predictive_cov_image_block.diag()).item()**0.5, torch.max(predictive_cov_image_block.diag()).item()**0.5)


        err = torch.abs(recon - example_image)
        print('error min, mean, max: ', torch.min(err).item(), torch.mean(err).item(), torch.max(err).item())

        predictive_cov_image_exact = get_exact_predictive_cov_image_mat(ray_trafos, bayesianized_model, jac, log_noise_model_variance_obs, eps=1e-6, cov_image_eps=1e-6)
        print('exact diag min, mean, max: ', torch.min(predictive_cov_image_exact.diag()).item(), torch.mean(predictive_cov_image_exact.diag()).item(), torch.max(predictive_cov_image_exact.diag()).item())
        print('exact diag min**0.5, mean**0.5, max**0.5: ', torch.min(predictive_cov_image_exact.diag()).item()**0.5, torch.mean(predictive_cov_image_exact.diag()).item()**0.5, torch.max(predictive_cov_image_exact.diag()).item()**0.5)

        ray_trafo_mat = torch.from_numpy(ray_trafos['ray_trafo_mat'])
        ray_trafo_mat = ray_trafo_mat.view(ray_trafo_mat.shape[0] * ray_trafo_mat.shape[1], -1).to(reconstructor.device)
        jac_obs = ray_trafo_mat.cuda() @ jac
        _, pred_test, _ = image_space_lin_model_post_pred_cov(block_priors, jac, jac_obs, torch.exp(log_noise_model_variance_obs))

        print('diff (predictive_cov_image_block-predictive_cov_image_exact).diag()', torch.sum(torch.abs(predictive_cov_image_block.diag() - predictive_cov_image_exact.diag())))

        lik_hess_inv_diag_mean = None
        if cfg.name in ['mnist', 'kmnist']:
            from dataset import extract_trafos_as_matrices
            import tensorly as tl
            tl.set_backend('pytorch')
            # pseudo-inverse computation
            trafos = extract_trafos_as_matrices(ray_trafos)
            trafo = trafos[0]
            if cfg.use_double:
                trafo = trafo.to(torch.float64)
            trafo = trafo.to(reconstructor.device)
            trafo_T_trafo = trafo.T @ trafo
            U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
            lik_hess_inv_diag_mean = (Vh.T @ torch.diag(1/S) @ U.T * torch.exp(log_noise_model_variance_obs)).diag().mean()
        elif cfg.name == 'walnut':
            # pseudo-inverse computation
            trafo = ray_trafos['ray_trafo_mat'].reshape(-1, np.prod(ray_trafos['space'].shape))
            if cfg.use_double:
                trafo = trafo.astype(np.float64)
            U_trafo, S_trafo, Vh_trafo = scipy.sparse.linalg.svds(trafo, k=100)
            # (Vh.T S U.T U S Vh)^-1 == (Vh.T S^2 Vh)^-1 == Vh.T S^-2 Vh
            S_inv_Vh_trafo = scipy.sparse.diags(1/S_trafo) @ Vh_trafo
            # trafo_T_trafo_inv_diag = np.diag(S_inv_Vh_trafo.T @ S_inv_Vh_trafo)
            trafo_T_trafo_inv_diag = np.sum(S_inv_Vh_trafo**2, axis=0)
            lik_hess_inv_diag_mean = np.mean(trafo_T_trafo_inv_diag) * np.exp(log_noise_model_variance_obs.item())
        print('noise_x_correction_term:', lik_hess_inv_diag_mean)

        if lik_hess_inv_diag_mean is not None:
            pred_test[np.diag_indices(pred_test.shape[0])] += lik_hess_inv_diag_mean
            predictive_cov_image_exact[np.diag_indices(predictive_cov_image_exact.shape[0])] += lik_hess_inv_diag_mean

        test_approx_blocks = True
        test_approx_blocks_from_samples = False
        test_from_samples_only_diag = False

        block_size_list = [cfg.size, 14, 7, 4, 2]

        approx_log_prob_list = []
        for block_size in block_size_list if test_approx_blocks else []:
            approx_log_prob, _, _, _, _ = predictive_image_log_prob(
                    recon.to(reconstructor.device), example_image.to(reconstructor.device),
                    ray_trafos, bayesianized_model, filtbackproj.to(reconstructor.device), reconstructor.model,
                    fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                    eps_mode='auto', eps=1e-3, cov_image_eps=1e-6,
                    block_size=block_size,
                    vec_batch_size=cfg.mrglik.impl.vec_batch_size,
                    cov_obs_mat_chol=cov_obs_mat_chol,
                    noise_x_correction_term=lik_hess_inv_diag_mean)
            approx_log_prob_list.append(approx_log_prob)

        num_mc_samples = 10000

        if test_approx_blocks_from_samples or test_from_samples_only_diag:
            mc_sample_images = sample_from_posterior(ray_trafos, observation.to(reconstructor.device), filtbackproj.to(reconstructor.device),
                    cov_obs_mat_chol, reconstructor.model, bayesianized_model, fwAD_be_model, fwAD_be_modules,
                    mc_samples=num_mc_samples, vec_batch_size=cfg.mrglik.impl.vec_batch_size)

        approx_log_prob_from_samples_list = []
        for block_size in block_size_list if test_approx_blocks_from_samples else []:
            approx_log_prob_from_samples, _, _, _, _ = predictive_image_log_prob_from_samples(
                    recon.to(reconstructor.device), observation.to(reconstructor.device), example_image.to(reconstructor.device),
                    ray_trafos, bayesianized_model, filtbackproj.to(reconstructor.device), reconstructor.model,
                    fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs,
                    eps_mode='auto', eps=1e-3,
                    block_size=block_size,
                    vec_batch_size=cfg.mrglik.impl.vec_batch_size,
                    cov_obs_mat_chol=cov_obs_mat_chol,
                    mc_sample_images=mc_sample_images,
                    noise_x_correction_term=lik_hess_inv_diag_mean)
            approx_log_prob_from_samples_list.append(approx_log_prob_from_samples)

        dist_block_priors = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=recon.to(reconstructor.device).flatten(),
                    scale_tril=torch.linalg.cholesky(pred_test)
                )
        log_prob_block_priors = dist_block_priors.log_prob(example_image.to(reconstructor.device).flatten())

        dist = torch.distributions.multivariate_normal.MultivariateNormal(
                    loc=recon.to(reconstructor.device).flatten(),
                    scale_tril=torch.linalg.cholesky(predictive_cov_image_exact)
                )
        log_prob = dist.log_prob(example_image.to(reconstructor.device).flatten())

        for block_size, approx_log_prob in zip(block_size_list, approx_log_prob_list):
            print('approx using block size {}:'.format(block_size), approx_log_prob / example_image.numel())

        for block_size, approx_log_prob_from_samples in zip(block_size_list, approx_log_prob_from_samples_list):
            print('approx from samples using block size {}:'.format(block_size), approx_log_prob_from_samples / example_image.numel())

        if test_from_samples_only_diag:
            log_prob_from_samples = approx_density_from_samples(recon.to(reconstructor.device), example_image.to(reconstructor.device), mc_sample_images, noise_x_correction_term=lik_hess_inv_diag_mean)

            print('sample based using only diag:', log_prob_from_samples / example_image.numel())

        print('exact using block_priors:', log_prob_block_priors / example_image.numel())
        print('exact:', log_prob / example_image.numel())

    print('max GPU memory used:', torch.cuda.max_memory_allocated() / example_image.numel())

if __name__ == '__main__':
    coordinator()
