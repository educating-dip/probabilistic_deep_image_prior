import hydra
import torch
import tqdm
import time
import numpy as np
import torch.nn as nn
from torch import linalg
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor, list_norm_layers, tv_loss
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, est_lin_var, sigmoid_gaussian_flow_log_prob, sigmoid_gaussian_exp, gaussian_log_prob
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

    for _ in range(10):
        examples = enumerate(loader)
        _, (example_image, _) = next(examples) # normalized between 0, 1

        # simulate and reconstruct the example image
        observation, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)
        dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 'reco_space': ray_trafos['space']}
        reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)

        # reconstruction - learning MAP estimate weights
        filtbackproj = filtbackproj.to(reconstructor.device)
        cfg.mrglik.optim.scl_fct_gamma = observation.view(-1).shape[0]

        recon, recon_no_sigmoid = reconstructor.reconstruct(observation, filtbackproj, example_image)
        recon_proj = ray_trafos['ray_trafo_module'](torch.from_numpy(recon).view(1, 1, 28, 28))

        lin_weights = None
        if cfg.linearize_weights:
            lin_weights, lin_pred = weights_linearization(cfg, filtbackproj, observation, example_image, reconstructor, ray_trafos)

        # estimate lik-noise precision
        mode = 'sigmoid_exact' if cfg.net.arch.use_sigmoid else 'linear'
        noise_mat, noise_mat_inv, noise_mat_det = est_lik_hess(observation, torch.from_numpy(recon), torch.from_numpy(recon_no_sigmoid), filtbackproj, ray_trafos, mode=mode)

        # estimate the Jacobian
        Jac = compute_jacobian_single_batch(filtbackproj, reconstructor.model, example_image.view(-1).shape[0])

        # opt marginal likelihood (type-II)
        block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, cfg.mrglik.priors.lengthscale_init, lin_weights=lin_weights)

        # compute variance pre-marginal likelihood optimisation (lengthscale & marginal prior variance)
        cov_diag_pre_mrglik, cov_pre_mrglik = est_lin_var(block_priors, Jac, noise_mat_inv)
        log_lik_pre_MLL_optim = sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(), cov_pre_mrglik, noise_mat_inv)
        print('log_lik pre marginal likelihood optim: {}'.format(log_lik_pre_MLL_optim))

        lik_variance = optim_marginal_lik(
            cfg,
            observation,
            recon_proj,
            filtbackproj,
            reconstructor,
            block_priors,
            Jac,
            noise_mat_det,
            noise_mat_inv,
            lin_weights,
            )

        (cov_diag_post, cov_post) = est_lin_var(block_priors, Jac,
                lik_variance*noise_mat_inv)

        log_lik_post_MLL_optim = \
            sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                                      cov_post, lik_variance*noise_mat_inv)

        baseline_variance = torch.mean((observation - recon_proj)**2)

        log_lik_noise_model = sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                            None, baseline_variance*noise_mat_inv)

        log_lik_noise_model_MLL_var = sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                            None, lik_variance*noise_mat_inv)

        print('test_log_lik post marginal likelihood optim: {}'.format(log_lik_post_MLL_optim))

        print('test_log_lik likelihood baseline: {}'.format(log_lik_noise_model))

        print('test_log_lik likelihood baseline (w optim lik_var): {}'.format(log_lik_noise_model_MLL_var))

        print('safety-check!')
        cfg.mrglik.optim.include_predCP = False
        block_priors = BlocksGPpriors(reconstructor.model,
                                      reconstructor.device,
                                      cfg.mrglik.priors.lengthscale_init,
                                      lin_weights=lin_weights)

        lik_variance = optim_marginal_lik(
            cfg,
            observation,
            recon_proj,
            filtbackproj,
            reconstructor,
            block_priors,
            Jac,
            noise_mat_det,
            noise_mat_inv,
            lin_weights,
            )

        (cov_diag_post_no_predCP, cov_post_no_predCP) = \
            est_lin_var(block_priors, Jac, lik_variance*noise_mat_inv)

        log_lik_no_PredCP = \
            sigmoid_gaussian_flow_log_prob(example_image.flatten(), torch.from_numpy(recon_no_sigmoid).flatten(),
                                      cov_post_no_predCP, lik_variance*noise_mat_inv)

        print('log_lik post marginal likelihood optim (no PredCP): {}'.format(log_lik_no_PredCP))

        dict = {'cfgs': cfg,
                    'test_log_lik':{
                        'before_MLL_optim': log_lik_pre_MLL_optim.item(),
                        'after_MLL_optim': log_lik_post_MLL_optim.item(),
                        'log_lik_noise_model': log_lik_noise_model.item(),
                        'log_lik_noise_model_MLL_var': log_lik_noise_model_MLL_var.item()},
                }

        data = {
                'recon': recon,
                'recon_lin_pred': lin_pred.cpu().numpy(),
                'noise_model_cov': noise_mat_inv.numpy(),
                'cov_post': cov_post.numpy(),
                'cov_post_no_predCP': cov_post_no_predCP.numpy()
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
