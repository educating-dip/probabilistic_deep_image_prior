import os
import hydra
import torch
import numpy as np
import random
import tensorly as tl
tl.set_backend('pytorch')
import torch.nn as nn
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from dataset import extract_trafos_as_matrices
from deep_image_prior import DeepImagePriorReconstructor
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, image_space_lin_model_post_pred_cov, gaussian_log_prob
from contextlib import redirect_stdout
from deep_image_prior.utils import PSNR, SSIM
from linearized_weights import weights_linearization

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    np.random.seed(cfg.net.torch_manual_seed)
    random.seed(cfg.net.torch_manual_seed)

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError

    ray_trafos = get_standard_ray_trafos(cfg, return_op_mat=True)
    test_log_lik_no_predcp_list, test_log_lik_noise_model_unit_var_list, \
         test_log_lik_noise_model_list, test_log_lik_predcp_list  = [], [], [], []
    # filename = os.path.join(os.getcwd(), 'results.txt')
    # with open(filename, 'w') as f:
    #     with redirect_stdout(f):
    for i, (example_image, _) in enumerate(loader):

        cfg.mrglik.optim.include_predcp = False
        # simulate and reconstruct the example image
        observation, filtbackproj, example_image = simulate(
            example_image, 
            ray_trafos, 
            cfg.noise_specs
            )
        dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 
                            'reco_space': ray_trafos['space']}
        reconstructor = DeepImagePriorReconstructor(
            **dip_ray_trafo, 
            cfg=cfg.net
            )
        # reconstruction - learning MAP estimate weights
        filtbackproj = filtbackproj.to(reconstructor.device)
        cfg.mrglik.optim.scl_fct_gamma = observation.view(-1).shape[0]
        recon, recon_no_sigmoid = reconstructor.reconstruct(
            observation, 
            filtbackproj, 
            example_image
            )
        bayesianized_model = BayesianizeModel(
                reconstructor, **{
                    'lengthscale_init': cfg.mrglik.priors.lengthscale_init,
                    'variance_init': cfg.mrglik.priors.variance_init},
                    include_normal_priors=cfg.mrglik.priors.include_normal_priors)
        all_modules_under_prior = bayesianized_model.get_all_modules_under_prior()
        torch.save(reconstructor.model.state_dict(),
                './reconstructor_model_{}.pt'.format(i))
        # estimate the Jacobian
        Jac = compute_jacobian_single_batch(
            filtbackproj,
            reconstructor.model, 
            all_modules_under_prior,
            example_image.flatten().shape[0]
            )
        trafos = extract_trafos_as_matrices(ray_trafos)
        trafo = trafos[0]
        proj_recon = trafo @ recon.flatten()
        Jac_obs = trafo.cuda() @ Jac

        # opt-marginal-likelihood w/o predcp
        if cfg.linearize_weights:
            linearized_weights, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
        else:
            linearized_weights = None
            lin_pred = None

        block_priors = BlocksGPpriors(
            reconstructor.model,
            bayesianized_model,
            reconstructor.device,
            cfg.mrglik.priors.lengthscale_init,
            cfg.mrglik.priors.variance_init,
            lin_weights=linearized_weights
            )
        noise_model_variance_obs_space_no_predcp = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (torch.from_numpy(recon), proj_recon), 
            block_priors,
            Jac,
            Jac_obs, 
            comment = '_no_predcp_recon_num_' + str(i)
            )
        torch.save(block_priors.state_dict(), 
            './block_priors_no_predcp_{}.pt'.format(i))
        (_, model_post_cov_no_predcp, Kxx_no_predcp) = image_space_lin_model_post_pred_cov(
            block_priors,
            Jac,
            Jac_obs, 
            noise_model_variance_obs_space_no_predcp
            )
        # pseudo-inverse computation
        trafo_T_trafo = trafo.cuda().T @ trafo.cuda()
        U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100)
        lik_hess_inv_no_predcp = Vh.T @ torch.diag(1/S) @ U.T * noise_model_variance_obs_space_no_predcp \
            + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)
        # for baselines's sake  
        lik_hess_inv_unit_var = Vh.T @ torch.diag(1/S) @ U.T \
            + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)
        assert lik_hess_inv_no_predcp.diag().min() > 0 
        # computing test-loglik MLL
        test_log_lik_no_predcp = gaussian_log_prob(
            example_image.flatten().cuda(),
            torch.from_numpy(recon).flatten().cuda(),
            model_post_cov_no_predcp, 
            None
            )
        # baselines 
        test_log_lik_noise_model_unit_var = gaussian_log_prob(
            example_image.flatten().cuda(),
            torch.from_numpy(recon).flatten().cuda(),
            None, 
            lik_hess_inv_unit_var
            )

        test_log_lik_noise_model = gaussian_log_prob(
            example_image.flatten().cuda(),
            torch.from_numpy(recon).flatten().cuda(),
            None,
            lik_hess_inv_no_predcp
            )

        print('test_log_lik marginal likelihood optim (no_predcp): {}'\
            .format(test_log_lik_no_predcp), flush=True)
        print('test_log_lik likelihood baseline (unit var): {}'\
            .format(test_log_lik_noise_model_unit_var), flush=True)
        print('test_log_lik likelihood baseline: {}'\
            .format(test_log_lik_noise_model), flush=True)

        # storing  
        test_log_lik_no_predcp_list.append(
            test_log_lik_no_predcp.item())
        test_log_lik_noise_model_unit_var_list.append(
            test_log_lik_noise_model_unit_var.item())
        test_log_lik_noise_model_list.append(
            test_log_lik_noise_model.item())

        # type-II MAP 
        cfg.mrglik.optim.include_predcp = True
        block_priors = BlocksGPpriors(
            reconstructor.model,
            bayesianized_model,
            reconstructor.device,
            cfg.mrglik.priors.lengthscale_init,
            cfg.mrglik.priors.variance_init,
            lin_weights=linearized_weights)
        noise_model_variance_obs_space_predcp = optim_marginal_lik_low_rank(
            cfg,
            observation,
            (torch.from_numpy(recon), proj_recon), 
            block_priors,
            Jac,
            Jac_obs,
            comment = '_predcp_recon_num_' + str(i)
            )
        
        torch.save(block_priors.state_dict(), 
            './block_priors_w_predcp_{}.pt'.format(i))

        lik_hess_inv_predcp = Vh.T @ torch.diag(1/S) @ U.T * noise_model_variance_obs_space_predcp\
            + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)

        (_, model_post_cov_predcp, Kxx_predcp) = image_space_lin_model_post_pred_cov(
                block_priors,
                Jac, 
                Jac_obs,
                noise_model_variance_obs_space_predcp
                )

        test_log_lik_predcp = gaussian_log_prob(
                example_image.flatten().cuda(),
                torch.from_numpy(recon).flatten().cuda(),
                model_post_cov_predcp,
                None
                )

        print('log_lik post marginal likelihood optim: {}'\
            .format(test_log_lik_predcp), flush=True)
        test_log_lik_predcp_list.append(test_log_lik_predcp.item())

        dict = {'cfgs': cfg,
                'test_log_lik':{
                    'test_loglik_type-II-MAP': test_log_lik_predcp.item(),
                    'test_loglik_MLL': test_log_lik_no_predcp.item(),
                    'noise_model': test_log_lik_noise_model.item(),
                    'noise_model_unit_variance': test_log_lik_noise_model_unit_var.item()},
                }

        data = {'observation': observation.cpu().numpy(), 
                'filtbackproj': filtbackproj.cpu().numpy(), 
                'image': example_image.cpu().numpy(), 
                'recon': recon,
                'recon_no_sigmoid': recon_no_sigmoid, 
                'Jac': Jac.cpu().numpy(), 
                'Jac_obs': Jac_obs.cpu().numpy(),
                'noise_model_variance_obs_space_predcp': noise_model_variance_obs_space_predcp.item(), 
                'noise_model_variance_obs_space_no_predcp': noise_model_variance_obs_space_no_predcp.item(),
                'noise_model_no_predcp': lik_hess_inv_no_predcp.cpu().numpy(),
                'noise_model_w_predcp': lik_hess_inv_predcp.cpu().numpy(),
                'model_post_cov_no_predcp': model_post_cov_no_predcp.detach().cpu().numpy(),
                'model_post_cov_predcp': model_post_cov_predcp.detach().cpu().numpy(),
                'Kxx_no_predcp': Kxx_no_predcp.detach().cpu().numpy(),
                'Kxx_predcp': Kxx_predcp.detach().cpu().numpy()
                }

        np.savez('test_log_lik_info_{}'.format(i), **dict)
        np.savez('recon_info_{}'.format(i), **data)

        if i >= cfg.num_images:
            break
    
    print('\n****************************************\n')
    print('Bayes DIP MLL: {:.4f}+/-{:.4f}\nB1:{:.4f}+/-{:.4f}\nB2:{:.4f}+/-{:.4f}\nBayes DIP Type-II MAP:{:.4f}+/-{:.4f}'.format(
            np.mean(test_log_lik_no_predcp_list),
            np.std(test_log_lik_no_predcp_list)/np.sqrt(cfg.num_images + 1), 
            np.mean(test_log_lik_noise_model_unit_var_list),
            np.std(test_log_lik_noise_model_unit_var_list)/np.sqrt(cfg.num_images + 1),
            np.mean(test_log_lik_noise_model_list),
            np.std(test_log_lik_noise_model_list)/np.sqrt(cfg.num_images + 1),
            np.mean(test_log_lik_predcp_list),
            np.std(test_log_lik_predcp_list)/np.sqrt(cfg.num_images + 1)
    ), 
    flush=True)

if __name__ == '__main__':

    coordinator()