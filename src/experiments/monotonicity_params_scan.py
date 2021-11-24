import hydra
import torch
import time
import random
import numpy as np
from torch import linalg
from omegaconf import DictConfig
from dataset import simulate
from dataset.utils import load_testset_KMNIST_dataset, get_standard_ray_trafos
from deep_image_prior import DeepImagePriorReconstructor, list_norm_layers,  set_all_weights, get_weight_vec, tv_loss
from priors_marglik import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

def get_average_var_group_filter(model, norm_layers):
    var = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            param_ = param.view(-1, *param.shape[2:]).flatten(start_dim=1)
            var.append(param_.var(dim=1).mean(dim=0).item())
    return np.mean(var)

def plot_monotonicity_traces(len_vec, overall_tv_samples_list, filename):

    def moving_average(x, w=5):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    fs_m1 = 6  # for figure ticks
    fs = 10    # for regular figure text
    fs_p1 = 15 #  figure titles

    matplotlib.rc('font', size=fs)          # controls default text sizes
    matplotlib.rc('axes', titlesize=fs)     # fontsize of the axes title
    matplotlib.rc('axes', labelsize=fs)     # fontsize of the x and y labels
    matplotlib.rc('xtick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('ytick', labelsize=fs_m1)    # fontsize of the tick labels
    matplotlib.rc('legend', fontsize=fs_m1)    # legend fontsize
    matplotlib.rc('figure', titlesize=fs_p1)   # fontsize of the figure title
    matplotlib.rc('font', **{'family':'serif', 'serif': ['Palatino']})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

    w = 100 # fixed window size
    fig, axs = plt.subplots(1, 4, figsize=(7, 2))
    titles = ['Down$_1$', 'Down$_2$', 'Up$_1$', 'Up$_2$']
    for i, (ax, tv_samples_list) in enumerate(zip(axs.flatten(), overall_tv_samples_list)):

        mean = np.mean(np.asarray(tv_samples_list), axis=1)
        mean = moving_average(mean, w)
        standard_dev = np.std(np.asarray(tv_samples_list), axis=1)
        standard_dev = moving_average(standard_dev, w) / np.sqrt(np.asarray(tv_samples_list).shape[1])

        ax.plot(len_vec[:-w+1], mean, linewidth=2.5, color='#EC2215')
        ax.fill_between(len_vec[:-w+1], mean-standard_dev, mean+standard_dev, color='#EC2215', alpha = 0.5)
        ax.set_xscale('log')
        ax.set_title(titles[i], pad=5)
        ax.grid(0.3)
        ax.set_xlabel('$\ell$', fontsize=12)
        ax.set_ylabel('$\kappa$', fontsize=12)
    
    plt.tight_layout()
    fig.savefig(filename + '.pdf', bbox_inches='tight')
    fig.savefig(filename + '.png', dpi=600)

@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    np.random.seed(cfg.net.torch_manual_seed)
    random.seed(cfg.net.torch_manual_seed)

    loader = load_testset_KMNIST_dataset()
    ray_trafos = get_standard_ray_trafos(cfg)
    examples = enumerate(loader)
    _, (example_image, _) = next(examples)
    _, filtbackproj, example_image = simulate(example_image, ray_trafos, cfg.noise_specs)
    dip_ray_trafo = {'ray_trafo_module': ray_trafos['ray_trafo_module'], 'reco_space': ray_trafos['space']}
    reconstructor = DeepImagePriorReconstructor(**dip_ray_trafo, cfg=cfg.net)
    # recon = reconstructor.reconstruct(observation, filtbackproj, example_image)
    path_to_params = os.path.join('/media/chen/Res/dip_bayesian_ext/src/experiments', 
        'multirun/2021-11-13/15-19-39/0/reconstructor_model_0.pt') # hard-coded 
    reconstructor.model.load_state_dict(torch.load(path_to_params))
    filtbackproj = filtbackproj.to(reconstructor.device)
    
    sampling_blocks = {
        'down_0': reconstructor.model.down[0],
        'down_1': reconstructor.model.down[1],
        'up_0': reconstructor.model.up[0],
        'up_1': reconstructor.model.up[1],
        }
    overall_TV_samples_list = []
    for key, block in tqdm(sampling_blocks.items()):
        for name, param in block.named_parameters():
            print(name, param.shape)
        norm_layers = list_norm_layers(block)
        out = get_weight_vec(block, norm_layers)
        len_out = len(out)
        with torch.no_grad():
            dist_func = lambda x: linalg.norm(x, ord=2) # setting GP cov dist func
            len_vec = np.logspace(-2, 2, 1000)
            variance_init = get_average_var_group_filter(block, norm_layers)
            tv_samples_list, recon_samples_list, filter_samples_list = [], [], []
            for lengthscale_init in tqdm(len_vec, leave=False):
                kernel_size = 3 # we discard layers that have kernel_size of 1
                cov_kwards = {
                    'kernel_size': kernel_size,
                    'lengthscale_init': lengthscale_init,
                    'variance_init': variance_init,
                    'dist_func': dist_func,
                    'store_device': reconstructor.device
                    }
                cov_func = RadialBasisFuncCov(**cov_kwards)
                GPp = GPprior(cov_func, reconstructor.device)
                tv_samples = []
                recon_samples = []
                for _ in range(500):
                    samples = GPp.sample(shape=[len_out//kernel_size**2])
                    set_all_weights(block, norm_layers, samples.flatten())
                    recon = reconstructor.model.forward(filtbackproj)[0]
                    tv_samples.append(tv_loss(recon).item())
                    recon_samples.append(recon.squeeze().detach().cpu().numpy())
                tv_samples_list.append(tv_samples)
                recon_samples_list.append(np.mean(np.asarray(recon_samples), axis=0))
                filter_samples_list.append(samples[0].detach().cpu().numpy())
            set_all_weights(block, norm_layers, out) # reverting to map weights
            data = {
                    'tv_samples': np.asarray(tv_samples_list),
                    'exp_recon': np.asarray(recon_samples_list),
                    'filters_samples': np.asarray(filter_samples_list)
                    }
            np.savez(key + '.npz', *data)
            time.sleep(0.01)
        overall_TV_samples_list.append(tv_samples_list)
        
    plot_monotonicity_traces(len_vec, overall_TV_samples_list, filename = 'monotonicity_plot')

if __name__ == '__main__':
    coordinator()