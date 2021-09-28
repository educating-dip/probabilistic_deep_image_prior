import odl
import hydra
import torch
import tqdm
import time
import numpy as np
from torch import linalg
from torch.nn import DataParallel
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor, list_norm_layers, tv_loss
from priors import *

def set_all_weights(model, norm_layers, weights):
    """ set all NN weights """
    assert not isinstance(model, DataParallel)
    n_weights_all = 0
    for name, param in model.named_parameters():
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            n_weights = param.numel()
            param.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(param))
            n_weights_all += n_weights

def get_weight_vec(model, norm_layers):
    ws = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            ws.append(param.flatten())
    return torch.cat(ws)

def get_average_var_group_filter(model, norm_layers):
    var = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            param_ = param.view(-1, *param.shape[2:]).flatten(start_dim=1)
            var.append(param_.var(dim=1).mean(dim=0).item())
    return np.mean(var)

def plot_monotonicity_traces(len_vec, tv_samples_list, filename):

    def moving_average(x, w=5):
        return np.convolve(x, np.ones(w), 'valid') / w

    import matplotlib.pyplot as plt

    w = 100 # fixed window size
    fig, ax = plt.subplots(figsize=(6, 6))
    mean = np.mean(np.asarray(tv_samples_list), axis=1)
    mean = moving_average(mean, w)
    standard_dev = np.std(np.asarray(tv_samples_list), axis=1)
    standard_dev = moving_average(standard_dev, w) / np.sqrt(np.asarray(tv_samples_list).shape[1])
    plt.plot(len_vec[:-w+1], mean, linewidth=2.5, color='#EC2215')
    plt.fill_between(len_vec[:-w+1], mean-standard_dev, mean+standard_dev, color='#EC2215', alpha = 0.5)
    plt.xscale('log')
    ax.set_xlabel('$\ell$', fontsize=12)
    ax.set_ylabel('$\kappa$', fontsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    fig.savefig(filename + '.pdf', bbox_inches='tight')

def fig_subplots_wrapper(n_rows, n_cols, len_vec, samples_list, filename):

    import matplotlib.pyplot as plt

    def GetSpacedElements(array, numElems=4):
        out = array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]
        return out

    idx_list = list(range(0, len(samples_list)))
    idx = GetSpacedElements(np.asarray(idx_list), numElems=n_rows*n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5,5),  facecolor='w', edgecolor='k', constrained_layout=True)
    minmin = np.min(np.asarray(samples_list))
    maxmax = np.max(np.asarray(samples_list))
    for i, ax in zip(idx, axs.flatten()):
        im = ax.imshow(samples_list[i], vmin=minmin, vmax=maxmax, cmap='gray')
        ax.set_title('$\ell$: {:.4f}'.format(len_vec[i]), fontsize=10)
        ax.set_axis_off()
    cbar = fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.95)
    fig.savefig(filename + '.pdf')

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    mnist_loader = load_testset_MNIST_dataset()
    ray_trafos = get_standard_ray_trafos(cfg)
    examples = enumerate(mnist_loader)
    batchsize, (example_image, example_targets) = next(examples)
    observation, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)
    dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 'reco_space': ray_trafos['space']}
    reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)
    recon = reconstructor.reconstruct(observation, filtbackproj, example_image)
    filtbackproj = filtbackproj.to(reconstructor.device)
    sampling_blocks = {
        'down_0': reconstructor.model.down[0],
        'down_1': reconstructor.model.down[1],
        'up_0': reconstructor.model.up[0],
        'up_1': reconstructor.model.up[1],
        'inc': reconstructor.model.inc,
        }

    for key, block in tqdm.tqdm(sampling_blocks.items()):
        for name, param in block.named_parameters():
            print(name, param.shape)
        norm_layers = list_norm_layers(block)
        out = get_weight_vec(block, norm_layers)
        len_out = len(out)
        with torch.no_grad():
            dist_func = lambda x: linalg.norm(x, ord=2) # setting GP cov dist func
            len_vec = np.logspace(-2, 2, 1000)
            tv_samples_list, recon_samples_list, filter_samples_list = [], [], []
            for lengthscale_init in tqdm.tqdm(len_vec, leave=False):
                kernel_size = 3 # we discard layers that have kernel_size of 1
                cov_kwards = {
                    'kernel_size': kernel_size,
                    'lengthscale_init': lengthscale_init,
                    'variance_init': get_mean_var_filter(block, norm_layers),
                    'dist_func': dist_func,
                    }
                cov_func = RadialBasisFuncCov(**cov_kwards)
                GPp = GPprior(cov_func)
                tv_samples = []
                recon_samples = []
                for _ in range(1000):
                    samples = GPp.sample(shape=[len_out//kernel_size**2])
                    set_all_weights(block, norm_layers, samples.flatten())
                    recon = reconstructor.model.forward(filtbackproj)
                    tv_samples.append(tv_loss(recon).item())
                    recon_samples.append(recon.squeeze().detach().cpu().numpy())
                tv_samples_list.append(tv_samples)
                recon_samples_list.append(np.mean(np.asarray(recon_samples), axis=0))
                filter_samples_list.append(samples[0].numpy())
            set_all_weights(block, norm_layers, out) # reverting to map weights
            data = {
                    'tv_samples': np.asarray(tv_samples_list),
                    'exp_recon': np.asarray(recon_samples_list),
                    'filters_samples': np.asarray(filter_samples_list)
                    }
            np.savez(key + '.npz', *data)
            plot_monotonicity_traces(len_vec, tv_samples_list, filename = key)
            fig_subplots_wrapper(5, 5, len_vec, recon_samples_list, filename = key + '_recon')
            fig_subplots_wrapper(5, 5, len_vec, filter_samples_list, filename = key + '_filter')
            time.sleep(0.01)

if __name__ == '__main__':
    coordinator()
