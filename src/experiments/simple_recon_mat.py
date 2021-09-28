import hydra
import numpy as np
import os
from dataset.utils import load_testset_MNIST_dataset, get_standard_ray_trafos
from dataset.mnist import simulate
from omegaconf import DictConfig

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    mnist_loader = load_testset_MNIST_dataset()
    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True,
            return_op_mat=True)
    examples = enumerate(mnist_loader)
    batchsize, (example_image, example_targets) = next(examples)
    observation, filtbackproj, example_image = simulate(example_image,
            ray_trafos, cfg.noise_specs, return_numpy=True)
    op_mat_vec = np.reshape(ray_trafos['ray_trafo_mat'],
                            (observation.flatten().shape[0],
                            example_image.flatten().shape[0]))
    observation_noiseless = op_mat_vec @ example_image.flatten()
    observation_mat = np.reshape(observation_noiseless, observation.shape)
    op_mat_vec_adj = op_mat_vec.T
    backproj = op_mat_vec_adj @ observation_noiseless
    backproj_mat = np.reshape(backproj, example_image.shape)

    # simple viz.
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(6, 6),
        facecolor='w',
        edgecolor='k',
        constrained_layout=True,
        )
    axs = axs.flatten()
    axs[0].imshow(observation_mat)
    axs[0].set_title('Observation Y')
    axs[1].imshow(backproj_mat)
    axs[1].set_title('Backproj')
    axs[2].imshow(example_image)
    axs[2].set_title('Image X')
    fig.savefig('recos.pdf')

if __name__ == '__main__':
    coordinator()
