import os
import hydra
from omegaconf import DictConfig
from dataset import get_pretraining_dataset
import torch
from deep_image_prior import DeepImagePriorReconstructor
from pretraining import Trainer

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    dataset, ray_trafos = get_pretraining_dataset(cfg,
            return_ray_trafo_torch_module=True)

    obs_shape = dataset.space[0].shape
    im_shape = dataset.space[1].shape

    ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'],
                 'reco_space': dataset.space[1]
                 }

    if cfg.pretraining.get('torch_manual_seed_init_model', None) is not None:
        torch.random.manual_seed(cfg.pretraining.torch_manual_seed_init_model)

    reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)
    trn_ray_trafos = {}
    Trainer(model=reconstructor.model,
            ray_trafos=trn_ray_trafos,
            cfg=cfg.trn).train(dataset)


if __name__ == '__main__':
    coordinator()
