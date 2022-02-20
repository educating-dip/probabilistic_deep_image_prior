import os
from itertools import islice
import numpy as np
import random
import hydra
from omegaconf import DictConfig, OmegaConf
from dataset.utils import (
        get_standard_ray_trafos,
        load_testset_MNIST_dataset, load_testset_KMNIST_dataset,
        load_testset_walnut)
from dataset.mnist import simulate
import torch
from hydra.utils import get_original_cwd
from deep_image_prior import DeepImagePriorReconstructor
from deep_image_prior.utils import PSNR, SSIM
from priors_marglik import BayesianizeModel
from scalable_linearised_laplace import (
        add_batch_grad_hooks, get_unet_batch_ensemble, get_fwAD_model,
        get_image_block_masks, get_prior_cov_obs_mat, clamp_params)

### Merges the results from a complete set of runs of
### ``assemble_cov_obs_mat.py`` with sub-slicing (the runs are specified via
### `density.merge_cov_obs_mat_sub_slices.load_path_list`).

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    if cfg.use_double:
        torch.set_default_tensor_type(torch.DoubleTensor)

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

    assert cfg.density.merge_cov_obs_mat_sub_slices.load_path_list is not None, "no previous run paths specified (density.merge_cov_obs_mat_sub_slices.load_path_list)"

    load_path_list = cfg.density.merge_cov_obs_mat_sub_slices.load_path_list

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        obs_numel = np.prod(ray_trafos['ray_trafo'].range.shape)
        cov_obs_mat = torch.zeros(obs_numel, obs_numel, device=device)

        vec_batch_size = cfg.mrglik.impl.vec_batch_size

        inds_covered = []
        for path in load_path_list:
            load_cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
            assert load_cfg.mrglik.impl.vec_batch_size == vec_batch_size

            cov_obs_mat_sub_slice_dict = torch.load(os.path.join(path, 'cov_obs_mat_sub_slice_{}.pt'.format(i)), map_location=device)

            sub_slice_batches = cov_obs_mat_sub_slice_dict['sub_slice_batches']
            sub_slice_inds = np.concatenate(
                    [np.arange(i, min(i+vec_batch_size, obs_numel))
                     for i in np.array(range(0, obs_numel, vec_batch_size))[sub_slice_batches]])

            cov_obs_mat[sub_slice_inds, :] = cov_obs_mat_sub_slice_dict['cov_obs_mat_sub_slice']
            inds_covered += sub_slice_inds.tolist()

        assert np.array_equal(sorted(inds_covered), range(obs_numel))

        torch.save({'cov_obs_mat': cov_obs_mat}, './cov_obs_mat_{}.pt'.format(i))

if __name__ == '__main__':
    coordinator()
