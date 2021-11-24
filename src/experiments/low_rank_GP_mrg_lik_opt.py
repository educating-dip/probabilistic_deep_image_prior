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
from dataset import extract_tafo_as_matrix
from deep_image_prior import DeepImagePriorReconstructor
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, low_rank_GP_lin_model_post_pred_cov, gaussian_log_prob
from linearized_weights import weights_linearization
from contextlib import redirect_stdout


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

    test_log_lik_no_PredCP_list, test_log_lik_noise_model_unit_var_list, \
         test_log_lik_noise_model_list, test_log_lik_w_PredCP_list  = [], [], [], []
    filename = os.path.join(os.getcwd(), 'results.txt')
    with open(filename, 'w') as f:
        with redirect_stdout(f):
            for i, (example_image, _) in enumerate(loader):

                cfg.mrglik.optim.include_predCP = False

                # simulate and reconstruct the example image
                observation, filtbackproj, example_image = simulate(example_image, 
                    ray_trafos, cfg.noise_specs)
                dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 
                                'reco_space': ray_trafos['space']}
                reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)

                # reconstruction - learning MAP estimate weights
                filtbackproj = filtbackproj.to(reconstructor.device)
                cfg.mrglik.optim.scl_fct_gamma = observation.view(-1).shape[0]

                recon, recon_no_sigmoid = reconstructor.reconstruct(observation, 
                    filtbackproj, example_image)
                
                torch.save(reconstructor.model.state_dict(),
                        './reconstructor_model_{}.pt'.format(i))

                lin_weights = None
                if cfg.linearize_weights:
                    lin_weights, lin_pred = weights_linearization(
                        cfg, 
                        filtbackproj, 
                        observation, 
                        example_image,
                        reconstructor,
                        ray_trafos
                    )
                
                # estimate the Jacobian
                Jac_x = compute_jacobian_single_batch(
                    filtbackproj,
                    reconstructor.model, 
                    example_image.flatten().shape[0]
                    )
                
                trafo, _, _, _ = extract_tafo_as_matrix(ray_trafos)

                Jac_y = trafo.cuda() @ Jac_x

                # opt marginal likelihood (type-II)
                block_priors = BlocksGPpriors(
                    reconstructor.model,
                    reconstructor.device,
                    cfg.mrglik.priors.lengthscale_init,
                    lin_weights=lin_weights)

                noise_model_variance_y_no_PredCP, MLL = optim_marginal_lik_low_rank_GP(
                    cfg,
                    observation,
                    recon, 
                    block_priors,
                    Jac_x,
                    Jac_y,
                    ray_trafos, 
                    lin_weights,
                    comment = '_no_PredCP_recon_num_' + str(i)
                    )
                
                torch.save(block_priors.state_dict(), 
                    './block_priors_no_PredCP_{}.pt'.format(i))
                
                print('MLL: {:.4f}'\
                    .format(MLL))

                (_, model_post_cov_no_PredCP, Kxx_no_PredCP) = low_rank_GP_lin_model_post_pred_cov(
                    block_priors,
                    Jac_x,
                    Jac_y, 
                    noise_model_variance_y_no_PredCP
                    )

                # pseudo-inverse computation 

                trafo_T_trafo = trafo.cuda().T @ trafo.cuda()

                U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100)

                lik_hess_inv_no_PredCP = Vh.T @ torch.diag(1/S) @ U.T * noise_model_variance_y_no_PredCP \
                    + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)
                
                # for baselines's sake  
                lik_hess_inv_unit_var = Vh.T @ torch.diag(1/S) @ U.T \
                    + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)

                assert lik_hess_inv_no_PredCP.diag().min() > 0 

                # computing test-loglik MLL 

                test_log_lik_no_PredCP = gaussian_log_prob(
                    example_image.flatten().cuda(),
                    torch.from_numpy(recon).flatten().cuda(),
                    model_post_cov_no_PredCP, 
                    lik_hess_inv_no_PredCP
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
                    lik_hess_inv_no_PredCP
                    )


                print('test_log_lik marginal likelihood optim (no_PredCP): {}'\
                    .format(test_log_lik_no_PredCP), flush=True)
                
                print('test_log_lik likelihood baseline (unit var): {}'\
                    .format(test_log_lik_noise_model_unit_var), flush=True)

                print('test_log_lik likelihood baseline: {}'\
                    .format(test_log_lik_noise_model), flush=True)

                # savings 
                test_log_lik_no_PredCP_list.append(
                    test_log_lik_no_PredCP.item())
                test_log_lik_noise_model_unit_var_list.append(
                    test_log_lik_noise_model_unit_var.item())
                test_log_lik_noise_model_list.append(
                    test_log_lik_noise_model.item())
                
                # type-II MAP 
                cfg.mrglik.optim.include_predCP = True

                block_priors = BlocksGPpriors(
                    reconstructor.model,
                    reconstructor.device,
                    cfg.mrglik.priors.lengthscale_init,
                    lin_weights=lin_weights)

                noise_model_variance_y_w_PredCP, MLL_MAP = optim_marginal_lik_low_rank_GP(
                    cfg,
                    observation,
                    recon, 
                    block_priors,
                    Jac_x,
                    Jac_y,
                    ray_trafos, 
                    lin_weights, 
                    comment = '_w_PredCP_recon_num_' + str(i)
                    )
                
                print('MLL_MAP: {:.4f}'\
                    .format(MLL_MAP), flush=True)
                
                torch.save(block_priors.state_dict(), 
                    './block_priors_w_PredCP_{}.pt'.format(i))

                lik_hess_inv_w_PredCP = Vh.T @ torch.diag(1/S) @ U.T * noise_model_variance_y_w_PredCP\
                    + 5e-4 * torch.eye(U.shape[0], device=block_priors.store_device)

                (_, model_post_cov_w_PredCP, Kxx_w_PredCP) = \
                    low_rank_GP_lin_model_post_pred_cov(
                        block_priors,
                        Jac_x, 
                        Jac_y,
                        noise_model_variance_y_w_PredCP
                        )

                test_log_lik_w_PredCP = \
                    gaussian_log_prob(
                        example_image.flatten().cuda(),
                        torch.from_numpy(recon).flatten().cuda(),
                        model_post_cov_w_PredCP,
                        lik_hess_inv_w_PredCP
                        )


                print('log_lik post marginal likelihood optim: {}'\
                    .format(test_log_lik_w_PredCP), flush=True)

                test_log_lik_w_PredCP_list.append(test_log_lik_w_PredCP.item())

                dict = {'cfgs': cfg,
                        'test_log_lik':{
                            'type-II-MAP': test_log_lik_w_PredCP.item(),
                            'MLL': test_log_lik_no_PredCP.item(),
                            'noise_model': test_log_lik_noise_model.item(),
                            'noise_model_unit_variance': test_log_lik_noise_model_unit_var.item()},
                        'MLL': MLL, 
                        'MLL_MAP': MLL_MAP
                        }

                try: 
                    lin_pred = lin_pred.cpu().numpy()
                except:
                    lin_pred = None

                data = {'observation': observation.cpu().numpy(), 
                        'filtbackproj': filtbackproj.cpu().numpy(), 
                        'image': example_image.cpu().numpy(), 
                        'recon': recon,
                        'recon_no_sigmoid': recon_no_sigmoid, 
                        'Jac_x': Jac_x.cpu().numpy(), 
                        'Jac_y': Jac_y.cpu().numpy(),
                        'recon_lin_pred': lin_pred,
                        'noise_model_variance_y_PredCP': noise_model_variance_y_w_PredCP.item(), 
                        'noise_model_variance_y_no_PredCP': noise_model_variance_y_no_PredCP.item(),
                        'noise_model_no_PredCP': lik_hess_inv_no_PredCP.cpu().numpy(),
                        'noise_model_w_PredCP': lik_hess_inv_w_PredCP.cpu().numpy(),
                        'model_post_cov_no_PredCP': model_post_cov_no_PredCP.detach().cpu().numpy(),
                        'model_post_cov_w_PredCP': model_post_cov_w_PredCP.detach().cpu().numpy(),
                        'Kxx_no_PredCP': Kxx_no_PredCP.detach().cpu().numpy(),
                        'Kxx_w_PredCP': Kxx_w_PredCP.detach().cpu().numpy()
                        }

                np.savez('test_log_lik_info_{}'.format(i), **dict)
                np.savez('recon_info_{}'.format(i), **data)

                if i >= cfg.num_images:
                    break
            
            print('\n****************************************\n')
            print('Bayes DIP MLL: {:.4f}+/-{:.4f}\nB1:{:.4f}+/-{:.4f}\nB2:{:.4f}+/-{:.4f}\nBayes DIP Type-II MAP:{:.4f}+/-{:.4f}'.format(
                    np.mean(test_log_lik_no_PredCP_list),
                    np.std(test_log_lik_no_PredCP_list) / np.sqrt(cfg.num_images + 1), 
                    np.mean(test_log_lik_noise_model_unit_var_list),
                    np.std(test_log_lik_noise_model_unit_var_list) / np.sqrt(cfg.num_images + 1),
                    np.mean(test_log_lik_noise_model_list),
                    np.std(test_log_lik_noise_model_list) / np.sqrt(cfg.num_images + 1),
                    np.mean(test_log_lik_w_PredCP_list),
                    np.std(test_log_lik_w_PredCP_list) / np.sqrt(cfg.num_images + 1)
            ), 
            flush=True)

if __name__ == '__main__':

    coordinator()