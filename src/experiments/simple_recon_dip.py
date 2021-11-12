import hydra
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.name == 'mnist':
        loader = load_testset_MNIST_dataset()
    elif cfg.name == 'kmnist':
        loader = load_testset_KMNIST_dataset()
    else:
        raise NotImplementedError
    
    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True)
    examples = enumerate(loader)
    _, (example_image, _) = next(examples)
    observation, filtbackproj, example_image = \
        simulate(example_image, ray_trafos, cfg.noise_specs)
    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': ray_trafos['space']}
    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)
    recon, _ = reconstructor.reconstruct(observation, filtbackproj, example_image)

    # simple viz.
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(
        1,
        4,
        figsize=(6, 6),
        facecolor='w',
        edgecolor='k',
        constrained_layout=True,
        )
    axs = axs.flatten()
    axs[0].imshow(observation[0, 0].numpy())
    axs[0].set_title('Observation Y')
    axs[1].imshow(filtbackproj[0, 0].numpy())
    axs[1].set_title('Filtbackproj')
    axs[2].imshow(example_image[0, 0].numpy())
    axs[2].set_title('Image X')
    axs[3].imshow(recon)
    axs[3].set_title('Recon')
    fig.savefig('recos.png')

if __name__ == '__main__':
    coordinator()
