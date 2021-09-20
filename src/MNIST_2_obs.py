import hydra
import numpy as np
import torch
import os
from dataset.utils import load_testset_MNIST_dataset, get_standard_ray_trafos
from dataset.mnist import simulate
from omegaconf import DictConfig

@hydra.main(config_path='cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    mnist_loader = load_testset_MNIST_dataset()
    ray_trafos = get_standard_ray_trafos(cfg)
    examples = enumerate(mnist_loader)
    batchsize, (example_image, example_targets) = next(examples)
    observation, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)

    # simple viz.
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(6, 6),  facecolor='w', edgecolor='k', constrained_layout=True)
    axs = axs.flatten()
    axs[0].imshow(observation[0, 0].numpy())
    axs[0].set_title('Observation Y')
    axs[1].imshow(filtbackproj[0, 0].numpy())
    axs[1].set_title('Filtbackproj')
    axs[2].imshow(example_image[0, 0].numpy())
    axs[2].set_title('Image X')
    plt.show()

if __name__ == '__main__':
    coordinator()
