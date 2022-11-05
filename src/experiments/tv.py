import os
from itertools import islice
import hydra
from omegaconf import DictConfig
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_rectangles_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
from scalable_bayes_exp_design.tvadam import TVAdamReconstructor
from deep_image_prior.utils import PSNR, SSIM

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True)

    # data: observation, filtbackproj, example_image
    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    elif cfg.name == 'rectangles':
        loader = load_testset_rectangles_dataset(cfg)
    elif cfg.name == 'walnut':
        loader = load_testset_walnut(cfg)
    else:
        raise NotImplementedError

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist', 'rectangles']:
            example_image = data_sample[0] if cfg.name in ['mnist', 'kmnist'] else data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            observation, filtbackproj, example_image = simulate(
                example_image,
                ray_trafos,
                cfg.noise_specs
                )
        else:
            raise NotImplementedError

        reconstructor = TVAdamReconstructor(ray_trafos['ray_trafo_module'], cfg=cfg.bed.tvadam)

        ray_trafos['ray_trafo_module'].to(reconstructor.device)
        ray_trafos['ray_trafo_module_adj'].to(reconstructor.device)

        recon = reconstructor.reconstruct(
                observation, fbp=filtbackproj.to(reconstructor.device), ground_truth=example_image.to(reconstructor.device),
                log=True)

        torch.save({'recon': torch.from_numpy(recon)[None, None]},
                './tv_recon_{}.pt'.format(i))

        print('DIP reconstruction of sample {:d}'.format(i))
        print('PSNR:', PSNR(recon, example_image[0, 0].cpu().numpy()))
        print('SSIM:', SSIM(recon, example_image[0, 0].cpu().numpy()))

if __name__ == '__main__':
    coordinator()
