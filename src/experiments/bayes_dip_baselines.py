from itertools import islice
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import tensorly as tl
tl.set_backend('pytorch')
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM, bayesianize_architecture, sample_from_bayesianized_model
from dataset import extract_trafos_as_matrices
from scalable_linearised_laplace import approx_density_from_samples

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
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            example_image, _ = data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
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

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        # pseudo-inverse computation
        trafos = extract_trafos_as_matrices(ray_trafos)
        trafo = trafos[0].to(reconstructor.device)
        trafo_T_trafo = trafo.T @ trafo
        U, S, Vh = tl.truncated_svd(trafo_T_trafo, n_eigenvecs=100) # costructing tsvd-pseudoinverse
        lik_hess_inv_no_predcp_diag_meam = (Vh.T @ torch.diag(1/S) @ U.T).diag().mean() # constructing noise in x correction term
        
        num_samples = 1000
        bayesianize_architecture(reconstructor.model, p=0.05)
        recon, _ = reconstructor.reconstruct(
                observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device), use_init_model=False)
        sample_recon = sample_from_bayesianized_model(reconstructor.model, filtbackproj.to(reconstructor.device), mc_samples=num_samples)
        mean = sample_recon.view(num_samples, -1).mean(dim=0)
        log_prob = approx_density_from_samples(mean, example_image.to(reconstructor.device), sample_recon, noise_x_correction_term=lik_hess_inv_no_predcp_diag_meam) / example_image.numel()

        torch.save(reconstructor.model.state_dict(),
                './dip_model_{}.pt'.format(i))

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

        data = {
                'filtbackproj': filtbackproj.cpu().numpy(), 
                'image': example_image.cpu().numpy(), 
                'recon': mean.cpu().numpy(),
                'test_log_likelihood': log_prob.cpu().numpy()
                }

        np.savez('recon_info_{}'.format(i), **data)

if __name__ == '__main__':
    coordinator()
