import os
from itertools import islice
import numpy as np
import random
import hydra
from omegaconf import DictConfig
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
        get_predictive_cov_image_block, predictive_image_block_log_prob,
        get_image_block_masks)

### Merges the results from a complete set of runs of
### ``compute_single_predictive_cov_block.py`` (specified via
### `density.merge_single_block_predictive_image_log_probs.load_path_list`).

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

    assert cfg.density.merge_single_block_predictive_image_log_probs.load_path_list is not None, "no previous run paths specified (density.merge_single_block_predictive_image_log_probs.load_path_list)"

    load_path_list = cfg.density.merge_single_block_predictive_image_log_probs.load_path_list

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    for i, _ in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        # merge predictive image log probs of single blocks in order to compute an approx. log prob of the full image
        block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

        load_paths_per_block = {}
        for load_path in load_path_list:
            # TODO assert cfg
            for block_idx in range(len(block_masks)):
                if os.path.isfile(os.path.join(load_path, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i))):
                    assert block_idx not in load_paths_per_block
                    load_paths_per_block[block_idx] = load_path

        block_diags = []
        block_log_probs = []
        block_mask_inds = []
        block_eps_values = []
        for block_idx, mask in enumerate(block_masks):
            load_path_block = load_paths_per_block[block_idx]
            predictive_image_log_prob_block_dict = torch.load(os.path.join(
                    load_path_block, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i)), map_location=device)
            block_diags.append(predictive_image_log_prob_block_dict['block_diag'])
            block_mask_inds.append(np.nonzero(mask)[0])
            block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'])
            block_eps_values.append(predictive_image_log_prob_block_dict['block_eps'])

        approx_log_prob = torch.sum(torch.stack(block_log_probs))

        torch.save({'approx_log_prob': approx_log_prob, 'block_mask_inds': block_mask_inds, 'block_log_probs': block_log_probs, 'block_diags': block_diags, 'block_eps_values': block_eps_values},
            './predictive_image_log_prob_{}.pt'.format(i))

        print('approx log prob ', approx_log_prob / np.prod(ray_trafos['space'].shape))


if __name__ == '__main__':
    coordinator()