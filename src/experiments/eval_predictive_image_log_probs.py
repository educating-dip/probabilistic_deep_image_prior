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
from scalable_linearised_laplace import get_image_block_mask_inds

### Evaluates the results from a set of runs of
### ``compute_single_predictive_cov_block.py`` or ``estimate_density_from_samples.py``
### (specified via `density.eval_predictive_image_log_probs.load_path_list`).

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

    assert cfg.density.eval_predictive_image_log_probs.load_path_list is not None, "no previous run paths specified (density.eval_predictive_image_log_probs.load_path_list)"

    load_path_list = cfg.density.eval_predictive_image_log_probs.load_path_list
    if isinstance(load_path_list, str):
        load_path_list = [load_path_list]

    device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    for i, data_sample in enumerate(islice(loader, cfg.num_images)):
        if i < cfg.get('skip_first_images', 0):
            continue

        if cfg.seed is not None:
            torch.manual_seed(cfg.seed + i)  # for reproducible noise in simulate

        if cfg.name in ['mnist', 'kmnist']:
            raise NotImplementedError  # load_path is missing

            example_image, _ = data_sample
            ray_trafos['ray_trafo_module'].to(example_image.device)
            ray_trafos['ray_trafo_module_adj'].to(example_image.device)
            if cfg.use_double:
                ray_trafos['ray_trafo_module'].to(torch.float64)
                ray_trafos['ray_trafo_module_adj'].to(torch.float64)
            observation, filtbackproj, example_image = simulate(
                example_image.double() if cfg.use_double else example_image, 
                ray_trafos,
                cfg.noise_specs
                )
            sample_dict = torch.load(os.path.join(load_path, 'sample_{}.pt'.format(i)), map_location=example_image.device)
            assert torch.allclose(sample_dict['filtbackproj'], filtbackproj)
            # filtbackproj = sample_dict['filtbackproj']
            # observation = sample_dict['observation']
            # example_image = sample_dict['ground_truth']
        elif cfg.name == 'walnut':
            observation, filtbackproj, example_image = data_sample
        else:
            raise NotImplementedError

        if cfg.use_double:
            observation = observation.double()
            filtbackproj = filtbackproj.double()
            example_image = example_image.double()

        reconstructor = DeepImagePriorReconstructor(**ray_trafo, cfg=cfg.net)

        if cfg.name in ['mnist', 'kmnist']:
            # model from previous run
            path = os.path.join(load_path, 'dip_model_{}.pt'.format(i))
        elif cfg.name == 'walnut':
            # fine-tuned model
            path = os.path.join(get_original_cwd(), reconstructor.cfg.finetuned_params_path 
                if reconstructor.cfg.finetuned_params_path.endswith('.pt') else reconstructor.cfg.finetuned_params_path + '.pt')
        else:
            raise NotImplementedError

        reconstructor.model.load_state_dict(torch.load(path, map_location=reconstructor.device))

        with torch.no_grad():
            reconstructor.model.eval()
            recon, _ = reconstructor.model.forward(filtbackproj.to(reconstructor.device))
        # recon = recon[0, 0].cpu().numpy()

        recon = recon.to(reconstructor.device)
        example_image = example_image.to(reconstructor.device)

        block_mask_inds = get_image_block_mask_inds(ray_trafos['space'].shape, block_size=cfg.density.block_size_for_approx, flatten=True)

        block_idx_list = cfg.density.compute_single_predictive_cov_block.block_idx  # may be used to restrict to a subset of blocks
        if block_idx_list is None:
            block_idx_list = list(range(len(block_mask_inds)))
        else:
            try:
                block_idx_list = list(block_idx_list)
            except TypeError:
                block_idx_list = [block_idx_list]

        def get_method_name(load_cfg, optim_cfg):
            method_name_parts = []
            method_name_parts.append('tv_map_{}'.format(optim_cfg.mrglik.optim.scaling_fct) if optim_cfg.mrglik.optim.include_predcp else 'mll')
            return '_'.join(method_name_parts)

        load_paths_per_block = {}
        for load_path in load_path_list:
            load_cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))
            load_job_name = OmegaConf.load(os.path.join(load_path, '.hydra', 'hydra.yaml')).hydra.job.name
            assert load_job_name in ['compute_single_predictive_cov_block', 'estimate_density_from_samples']
            assert load_cfg.density.block_size_for_approx == cfg.density.block_size_for_approx

            optim_path = load_cfg.density.compute_single_predictive_cov_block.load_path
            optim_cfg = OmegaConf.load(os.path.join(optim_path, '.hydra', 'config.yaml'))
            assert optim_cfg.linearize_weights == True
            method_name = get_method_name(load_cfg, optim_cfg)

            blocks_found = []
            for block_idx in block_idx_list:
                if os.path.isfile(os.path.join(load_path, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i))):
                    load_paths_per_block.setdefault(block_idx, {})
                    assert method_name not in load_paths_per_block[block_idx]
                    load_paths_per_block[block_idx][method_name] = load_path
                    blocks_found.append(block_idx)
            blocks_found = [block_idx for block_idx in load_cfg.density.compute_single_predictive_cov_block.block_idx if block_idx in blocks_found and block_idx in block_idx_list]  # sort
            blocks_not_found = [block_idx for block_idx in load_cfg.density.compute_single_predictive_cov_block.block_idx if block_idx not in blocks_found and block_idx in block_idx_list]
            print('in path {}:'.format(load_path))
            print('  found blocks {}'.format(blocks_found))
            print('  missing blocks {}'.format(blocks_not_found))

        # load_path_df = pd.DataFrame([load_paths for block_idx, load_paths in load_paths_per_block.items()])

        def get_metrics(block_idx, load_path_block):
            predictive_image_log_prob_block_dict = torch.load(os.path.join(
                    load_path_block, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i)), map_location=device)
            mask_inds = predictive_image_log_prob_block_dict['mask_inds']
            metrics = {
                'block_idx': block_idx,
                'log_prob': predictive_image_log_prob_block_dict['block_log_prob'].item() / example_image.flatten()[mask_inds].numel(),
                'mean_abs_error': torch.mean(torch.abs((recon.view(-1)-example_image.view(-1))[mask_inds])).item(),
                'max_abs_error': torch.max(torch.abs((recon.view(-1)-example_image.view(-1))[mask_inds])).item(),
                'min_sqrt_diag': torch.min(torch.sqrt(predictive_image_log_prob_block_dict['block_diag'])).item(),
                'block_eps': predictive_image_log_prob_block_dict['block_eps'],
            }
            return metrics

        metrics_records = []
        for block_idx, load_paths in load_paths_per_block.items():
            for method_name, load_path_block in load_paths.items():
                record = {'block_idx': block_idx, 'method_name': method_name}
                record.update(get_metrics(block_idx, load_path_block))
                metrics_records.append(record)

        with open('results_{}.json'.format(i), 'w') as f:
            json.dump(metrics_records, f, indent=1)

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
        common_blocks = np.array(range(len(block_mask_inds)))
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
