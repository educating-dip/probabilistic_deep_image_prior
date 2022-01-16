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
### `density.eval_predictive_image_log_probs.load_path_list`).

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
        block_eps_values = []
        # for block_idx, load_path_block in enumerate(load_path_list):
        for block_idx, mask in enumerate(block_masks):
            load_path_block = load_paths_per_block.get(block_idx)
            if load_path_block is None:
                continue
            predictive_image_log_prob_block_dict = torch.load(os.path.join(
                    load_path_block, 'predictive_image_log_prob_block{}_{}.pt'.format(block_idx, i)), map_location=device)
            block_diags.append(predictive_image_log_prob_block_dict['block_diag'])
            block_log_probs.append(predictive_image_log_prob_block_dict['block_log_prob'])
            block_eps_values.append(predictive_image_log_prob_block_dict['block_eps'])

            mask = predictive_image_log_prob_block_dict['mask']
            print('block_idx', block_idx)
            print('  log_prob', predictive_image_log_prob_block_dict['block_log_prob'].item() / example_image.flatten()[mask].numel())
            print('  mae', torch.mean(torch.abs((recon.view(-1)-example_image.view(-1))[mask])).item())
            print('  max abs error', torch.max(torch.abs((recon.view(-1)-example_image.view(-1))[mask])).item())
            print('  min sqrt(diag)', torch.min(torch.sqrt(predictive_image_log_prob_block_dict['block_diag'])).item())
            print('  block_eps', predictive_image_log_prob_block_dict['block_eps'])

if __name__ == '__main__':
    coordinator()
