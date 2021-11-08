import pdb
import hydra
import torch
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, compute_lin_pred_cov, sigmoid_gaussian_flow_log_prob
from linearized_weights import weights_linearization

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError

    ray_trafos = get_standard_ray_trafos(cfg, return_op_mat=True)

    for example_image, _ in loader:

        cfg.mrglik.optim.include_predCP = True

        # simulate and reconstruct the example image
        observation, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)
        dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 'reco_space': ray_trafos['space']}
        reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)

        # reconstruction - learning MAP estimate weights
        filtbackproj = filtbackproj.to(reconstructor.device)
        cfg.mrglik.optim.scl_fct_gamma = observation.view(-1).shape[0]

        recon, recon_no_sigmoid = reconstructor.reconstruct(observation, filtbackproj, example_image)

        lin_weights = None
        if cfg.linearize_weights:
            lin_weights, lin_pred = weights_linearization(cfg, filtbackproj, observation, example_image, reconstructor, ray_trafos)
        
        # estimate the Jacobian
        Jac = compute_jacobian_single_batch(filtbackproj, reconstructor.model, example_image.view(-1).shape[0])

        # opt marginal likelihood (type-II)
        block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, cfg.mrglik.priors.lengthscale_init, lin_weights=lin_weights)

        _, _, lik_hess_inv = optim_marginal_lik(
            cfg,
            observation,
            torch.from_numpy(recon), 
            torch.from_numpy(recon_no_sigmoid), 
            filtbackproj,
            block_priors,
            Jac,
            ray_trafos, 
            lin_weights,
            )

        (_, cov_post) = compute_lin_pred_cov(block_priors, Jac, lik_hess_inv)


        log_lik_post_MLL_optim = \
            sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                                      cov_post, lik_hess_inv)

        log_lik_noise_model_MLL_var = sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                            None, lik_hess_inv)

        print('test_log_lik post marginal likelihood optim: {}'.format(log_lik_post_MLL_optim))

        print('test_log_lik likelihood baseline (w optim lik_var): {}'.format(log_lik_noise_model_MLL_var))

        print('safety-check!')
        cfg.mrglik.optim.include_predCP = False
        block_priors = BlocksGPpriors(reconstructor.model,
                                      reconstructor.device,
                                      cfg.mrglik.priors.lengthscale_init,
                                      lin_weights=lin_weights)

        _, _, lik_hess_inv_no_PredCP = optim_marginal_lik(
            cfg,
            observation,
            torch.from_numpy(recon), 
            torch.from_numpy(recon_no_sigmoid), 
            filtbackproj,
            block_priors,
            Jac,
            ray_trafos, 
            lin_weights,
            )

        (_, cov_post_no_predCP) = \
            compute_lin_pred_cov(block_priors, Jac, lik_hess_inv_no_PredCP)

        log_lik_no_PredCP = \
            sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten().cuda(),
                                      cov_post_no_predCP, lik_hess_inv_no_PredCP)

        print('log_lik post marginal likelihood optim (no PredCP): {}'.format(log_lik_no_PredCP))

        dict = {'cfgs': cfg,
                    'test_log_lik':{
                        'MLL_optim': log_lik_post_MLL_optim.item(),
                        'log_lik_noise_model_MLL_var': log_lik_noise_model_MLL_var.item(),
                        'log_lik_no_PredCP': log_lik_no_PredCP.item()},
                }

        try: 
            lin_pred = lin_pred.cpu().numpy()
        except:
             lin_pred = None

        data = {
                'recon': recon,
                'recon_lin_pred': lin_pred,
                'noise_model_cov': lik_hess_inv.cpu().numpy(),
                'cov_post': cov_post.cpu().numpy(),
                'cov_post_no_predCP': cov_post_no_predCP.cpu().numpy()
                }

        np.savez('log_lik_info', **dict)
        np.savez('recon_info', **data)


# minmin = torch.min(torch.stack((cov_diag_pre, cov_diag_post)))
# maxmax = torch.max(torch.stack((cov_diag_pre, cov_diag_post)))
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# fig, axs = plt.subplots(3, 3, figsize=(20, 20))
# axs = axs.flatten()
# for img, ax, title in zip([cov_diag_pre, cov_diag_post], axs, ['pixel-wise var unit', 'pixel-wise var PredCP']):
#     im = ax.imshow(img.detach().cpu().numpy().reshape(28, 28), vmax = maxmax, vmin=minmin, cmap='gray')
#     ax.title.set_text(title)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax, orientation='vertical')
#
# axs[2].imshow(example_image[0, 0].numpy(), vmax=1, vmin=0, cmap='gray')
# axs[2].title.set_text('Image')
# axs[3].imshow(filtbackproj[0, 0].detach().cpu().numpy(), vmax=1, vmin=0, cmap='gray')
# axs[3].title.set_text('FBP')
# axs[4].imshow(recon, vmax=1, vmin=0, cmap='gray')
# axs[4].title.set_text('Reco')
# im = axs[5].imshow( (example_image[0, 0].numpy() - recon)**2, cmap='gray')
# print(np.mean((example_image[0, 0].flatten().numpy() - recon.flatten())**2))
# axs[5].title.set_text('SE')
# divider = make_axes_locatable(axs[5])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax, orientation='vertical')
#
# im = axs[6].imshow((example_image[0, 0].numpy() - recon_with_uq.detach().cpu().numpy().reshape(28, 28))**2, cmap='gray')
# print(np.mean((example_image[0, 0].flatten().numpy() - recon_with_uq.detach().cpu().numpy().flatten())**2))
# divider = make_axes_locatable(axs[6])
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(im, cax=cax, orientation='vertical')
# plt.savefig('uncertainty_analysis.pdf')

if __name__ == '__main__':
    coordinator()
