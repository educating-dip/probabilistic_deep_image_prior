from itertools import islice
import hydra
import torch
import numpy as np
import random
import tensorly as tl
tl.set_backend('pytorch')
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from dataset import extract_trafos_as_matrices
from deep_image_prior import DeepImagePriorReconstructor
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, image_space_lin_model_post_pred_cov, gaussian_log_prob
from deep_image_prior.utils import PSNR, SSIM
from linearized_weights import weights_linearization


def scale_up_weights(model, scale=100):
    for layer in model.up.modules():
        if isinstance(layer, torch.nn.Conv2d): 
            layer.weight.data *= scale 
            layer.bias.data *= scale  


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

    np.random.seed(cfg.net.torch_manual_seed)
    random.seed(cfg.net.torch_manual_seed)

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError

    ray_trafos = get_standard_ray_trafos(cfg, return_op_mat=True)
    test_log_lik_no_predcp_list = []
    for i, (example_image, _) in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        cfg.mrglik.optim.include_predcp = False
        # simulate and reconstruct the example image
        ray_trafos['ray_trafo_module'].to(example_image.device)
        ray_trafos['ray_trafo_module_adj'].to(example_image.device)
        if cfg.use_double:
            ray_trafos['ray_trafo_module'].to(torch.float64)
            ray_trafos['ray_trafo_module_adj'].to(torch.float64)
        observation, filtbackproj, example_image = simulate(
            example_image.double() if cfg.use_double else example_image, 
            ray_trafos, 
            cfg.noise_specs
            )
        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()
        dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 
                            'reco_space': ray_trafos['space']}
        reconstructor = DeepImagePriorReconstructor(
            **dip_ray_trafo, 
            cfg=cfg.net
            )
        # reconstruction - learning MAP estimate weights
        filtbackproj = filtbackproj.to(reconstructor.device)
        if cfg.load_dip_models_from_path is not None:
            path = os.path.join(cfg.load_dip_models_from_path, 'dip_model_{}.pt'.format(i))
            print('loading model for {} reconstruction from {}'.format(cfg.name, path))
            reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))
            with torch.no_grad():
                reconstructor.model.eval()
                recon, recon_no_sigmoid = reconstructor.model.forward(filtbackproj)
            recon = recon[0, 0].cpu().numpy()
        else:
            recon, recon_no_sigmoid = reconstructor.reconstruct(
                observation, fbp=filtbackproj, ground_truth=example_image)
            torch.save(reconstructor.model.state_dict(),
                    './dip_model_{}.pt'.format(i))
        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))
        bayesianized_model = BayesianizeModel(
                reconstructor, **{
                    'lengthscale_init': cfg.mrglik.priors.lengthscale_init,
                    'variance_init': cfg.mrglik.priors.variance_init},
                    include_normal_priors=cfg.mrglik.priors.include_normal_priors)
        all_modules_under_prior = bayesianized_model.get_all_modules_under_prior()
        # estimate the Jacobian
        Jac = compute_jacobian_single_batch(
            filtbackproj,
            reconstructor.model, 
            all_modules_under_prior,
            example_image.flatten().shape[0]
            )
        trafos = extract_trafos_as_matrices(ray_trafos)
        trafo = trafos[0]
        if cfg.use_double:
            trafo = trafo.to(torch.float64)
        proj_recon = trafo @ recon.flatten()
        Jac_obs = trafo.to(reconstructor.device) @ Jac

        # scaling up 
        # scale_up_weights(reconstructor.model, scale=100)
        # out = reconstructor.model.forward(filtbackproj.to(reconstructor.device))[0].detach().cpu().numpy()
        # print('PSNR:', PSNR(out[0, 0], example_image[0, 0].cpu().numpy()))
        # print('SSIM:', SSIM(out[0, 0], example_image[0, 0].cpu().numpy()))

        # opt-marginal-likelihood w/o predcp
        if cfg.linearize_weights:
            linearized_weights, lin_pred = weights_linearization(cfg, bayesianized_model, filtbackproj, observation, example_image, reconstructor, ray_trafos)
            print('linear reconstruction sample {:d}'.format(i))
            print('PSNR:', PSNR(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))
            print('SSIM:', SSIM(lin_pred[0, 0].cpu().numpy(), example_image[0, 0].cpu().numpy()))

            torch.save({'linearized_weights': linearized_weights, 'linearized_prediction': lin_pred},  
                './linearized_weights_{}.pt'.format(i))
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

        torch.save(bayesianized_model.state_dict(), 
            './bayesianized_model_no_predcp_{}.pt'.format(i))
        torch.save(block_priors.state_dict(), 
            './block_priors_no_predcp_{}.pt'.format(i))
        torch.save({'noise_model_variance_obs_space_no_predcp': noise_model_variance_obs_space_no_predcp},
            './noise_model_variance_obs_space_no_predcp_{}.pt'.format(i))

        (_, model_post_cov_no_predcp, Kxx_no_predcp) = image_space_lin_model_post_pred_cov(
            block_priors,
            Jac,
            Jac_obs, 
            noise_model_variance_obs_space_no_predcp
            )
        
        # computing test-loglik MLL
        model_post_cov_no_predcp[np.diag_indices(784)] += 0.000326
        test_log_lik_no_predcp = gaussian_log_prob(
            example_image.flatten().to(block_priors.store_device),
            torch.from_numpy(recon).flatten().to(block_priors.store_device),
            model_post_cov_no_predcp, 
            None
            )

        print('test_log_lik marginal likelihood optim (no_predcp): {}'\
            .format(test_log_lik_no_predcp), flush=True)

        # storing  
        test_log_lik_no_predcp_list.append(
            test_log_lik_no_predcp.item())

        dict = {
                'test_log_lik':{
                    'test_loglik_MLL': test_log_lik_no_predcp.item()},
                }

        data = {'observation': observation.cpu().numpy(), 
                'filtbackproj': filtbackproj.cpu().numpy(), 
                'image': example_image.cpu().numpy(), 
                'recon': recon,
                'recon_no_sigmoid': recon_no_sigmoid, 
                'noise_model_variance_obs_space_no_predcp': noise_model_variance_obs_space_no_predcp.item(),
                'model_post_cov_no_predcp': model_post_cov_no_predcp.detach().cpu().numpy(),
                'Kxx_no_predcp': Kxx_no_predcp.detach().cpu().numpy(),
                }

        np.savez('test_log_lik_info_{}'.format(i), **dict)
        np.savez('recon_info_{}'.format(i), **data)
    
    print('\n****************************************\n')
    print('Bayes DIP MLL: {:.4f}+/-{:.4f}\nB1:{:.4f}+/-{:.4f}\nB2:{:.4f}+/-{:.4f}\nBayes DIP Type-II MAP:{:.4f}+/-{:.4f}'.format(
            np.mean(test_log_lik_no_predcp_list),
            np.std(test_log_lik_no_predcp_list)/np.sqrt(cfg.num_images),
    ), 
    flush=True)

if __name__ == '__main__':

    coordinator()