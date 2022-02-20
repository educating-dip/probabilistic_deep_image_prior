import os
import json
from itertools import islice
import numpy as np
import pandas as pd
import random
import hydra
from omegaconf import OmegaConf, DictConfig
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
        get_image_block_masks)

### Evaluates the results from a set of runs of
### ``compute_single_predictive_cov_block.py`` (specified via
### `density.re_eval_predictive_image_log_probs.load_path`), optionally
### restricted to a subset of blocks (specified via
### density.re_eval_predictive_image_log_probs.block_idx``).

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

    load_path = cfg.density.re_eval_predictive_image_log_probs.load_path

    load_cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))
    assert load_cfg.name == cfg.name

    block_masks = get_image_block_masks(ray_trafos['space'].shape, block_size=load_cfg.density.block_size_for_approx, flatten=True)
    block_idx_list = cfg.density.re_eval_predictive_image_log_probs.get('block_idx', list(range(len(block_masks))))
    try:
        block_idx_list = list(block_idx_list)
    except TypeError:
        block_idx_list = [block_idx_list]

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        with open(os.path.join(load_path, 'results_{}.json'.format(i)), 'r') as f:
            metrics_records = json.load(f)

        df = pd.DataFrame(metrics_records)
        df.set_index('block_idx')
        results_table_str = df.to_string()
        print(results_table_str)

        with open('results_{}.txt'.format(i), 'w') as f:
            f.write(results_table_str)

        log_probs_per_method = {}
        for record in metrics_records:
            log_probs_per_method.setdefault(record['method_name'], {})
            log_probs_per_method[record['method_name']][record['block_idx']] = record['log_prob']
        common_blocks = np.array(block_idx_list)
        for method_name, log_probs_dict in log_probs_per_method.items():
            common_blocks = np.intersect1d(common_blocks, list(log_probs_dict.keys()))
        print('number of common blocks for all methods:', len(common_blocks))
        mean_log_probs_per_method_on_common_blocks = {
            method_name: np.mean([log_probs_dict[block_idx] for block_idx in common_blocks])
            for method_name, log_probs_dict in log_probs_per_method.items()
        }
        print('mean log probs for common blocks:', mean_log_probs_per_method_on_common_blocks)

if __name__ == '__main__':
    coordinator()
