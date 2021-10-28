import hydra
import torch
import tqdm
import time
import numpy as np
import torch.nn as nn
from torch import linalg
from torch.nn import DataParallel
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor, list_norm_layers, tv_loss
from priors_marglik import *
from linearized_laplace import compute_jacobian_single_batch, est_lin_var, sigmoid_guassian_log_prob, sigmoid_gaussian_exp

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError

    ray_trafos = get_standard_ray_trafos(cfg, return_op_mat=True)
    examples = enumerate(loader)
    _, (example_image, _) = next(examples) # normalize between 0, 1

    # simulate and reconstruct the example image
    observation, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)
    dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 'reco_space': ray_trafos['space']}
    reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)
    # reconstruction - learning MAP estimate weights
    filtbackproj = filtbackproj.to(reconstructor.device)
    cfg.mrglik.optim.scl_fct_gamma = observation.view(-1).shape[0]
    recon, recon_no_sigmoid = reconstructor.reconstruct(observation, filtbackproj, example_image)

    # estimate lik-noise precision
    mode = 'sigmoid_exact' if cfg.net.arch.use_sigmoid else 'linear'
    noise_mat, noise_mat_inv, noise_mat_det = estimate_lik_noise(observation, filtbackproj, reconstructor, ray_trafos, mode=mode)

    # estimate the Jacobian
    Jac = compute_jacobian_single_batch(filtbackproj, reconstructor.model, example_image.view(-1).shape[0])

    # opt marginal likelihood (type-II)
    block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, cfg.mrglik.priors.lengthscale_init)

    # compute variance pre-marginal likelihood optimisation (lengthscale & marginal prior variance)
    cov_diag_pre, cov_pre = est_lin_var(block_priors, Jac, noise_mat_inv, return_numpy=False)
    log_lik = sigmoid_guassian_log_prob(example_image.flatten().to(reconstructor.device), torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_pre)
    print('log_lik pre marginal likelihood optim: {}'.format(log_lik))

    recon_with_uq = sigmoid_gaussian_exp(torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_pre)
    mse = torch.mean((example_image[0, 0].flatten().to(reconstructor.device) - recon_with_uq.flatten())**2)
    print(mse)

    optim_marginal_lik(cfg, reconstructor, filtbackproj, block_priors, Jac, noise_mat_det, noise_mat_inv, reconstructor.device)

    cov_diag_post, cov_post = est_lin_var(block_priors, Jac, noise_mat_inv, return_numpy=False)
    log_lik = sigmoid_guassian_log_prob(example_image.flatten().to(reconstructor.device), torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_post)
    print('log_lik post marginal likelihood optim: {}'.format(log_lik))

    recon_with_uq = sigmoid_gaussian_exp(torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_post)
    mse = torch.mean((example_image[0, 0].flatten().to(reconstructor.device) - recon_with_uq.flatten())**2)
    print(mse)


    print('safety-check!')
    cfg.mrglik.optim.include_predCP = False
    block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, cfg.mrglik.priors.lengthscale_init)

    optim_marginal_lik(cfg, reconstructor, filtbackproj, block_priors, Jac, noise_mat_det, noise_mat_inv, reconstructor.device)

    cov_diag_post_no_predCP, cov_post_no_predCP = est_lin_var(block_priors, Jac, noise_mat_inv, return_numpy=False)

    log_lik = sigmoid_guassian_log_prob(example_image.flatten().to(reconstructor.device), torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_post_no_predCP)
    print('log_lik post marginal likelihood optim (no PredCP): {}'.format(log_lik))

    recon_with_uq = sigmoid_gaussian_exp(torch.from_numpy(recon_no_sigmoid).flatten().to(reconstructor.device), cov_post)
    mse = torch.mean((example_image[0, 0].flatten().to(reconstructor.device) - recon_with_uq.flatten())**2)
    print(mse)

    minmin = torch.min(torch.stack((cov_diag_pre, cov_diag_post)))
    maxmax = torch.max(torch.stack((cov_diag_pre, cov_diag_post)))

    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axs = plt.subplots(3, 3, figsize=(20, 20))
    axs = axs.flatten()
    for img, ax, title in zip([cov_diag_pre, cov_diag_post], axs, ['pixel-wise var unit', 'pixel-wise var PredCP']):
        im = ax.imshow(img.detach().cpu().numpy().reshape(28, 28), vmax = maxmax, vmin=minmin, cmap='gray')
        ax.title.set_text(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    axs[2].imshow(example_image[0, 0].numpy(), vmax=1, vmin=0, cmap='gray')
    axs[2].title.set_text('Image')
    axs[3].imshow(filtbackproj[0, 0].detach().cpu().numpy(), vmax=1, vmin=0, cmap='gray')
    axs[3].title.set_text('FBP')
    axs[4].imshow(recon, vmax=1, vmin=0, cmap='gray')
    axs[4].title.set_text('Reco')
    im = axs[5].imshow( (example_image[0, 0].numpy() - recon)**2, cmap='gray')
    print(np.mean((example_image[0, 0].flatten().numpy() - recon.flatten())**2))
    axs[5].title.set_text('SE')
    divider = make_axes_locatable(axs[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    im = axs[6].imshow((example_image[0, 0].numpy() - recon_with_uq.detach().cpu().numpy().reshape(28, 28))**2, cmap='gray')
    print(np.mean((example_image[0, 0].flatten().numpy() - recon_with_uq.detach().cpu().numpy().flatten())**2))
    divider = make_axes_locatable(axs[6])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig('uncertainty_analysis.pdf')

if __name__ == '__main__':
    coordinator()
