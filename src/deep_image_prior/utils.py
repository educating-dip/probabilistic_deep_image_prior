import torch
import numpy as np
from skimage.metrics import structural_similarity
from torch.nn import DataParallel
from collections.abc import Iterable

def set_all_weights(model, norm_layers, weights):
    """ set all NN weights """
    assert not isinstance(model, DataParallel)
    n_weights_all = 0
    for name, param in model.named_parameters():
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            n_weights = param.numel()
            param.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(param))
            n_weights_all += n_weights

def set_all_weights_block(model, weights, include_block=['down', 'up']):
    """ set all NN weights """
    assert not isinstance(model, DataParallel)
    n_weights_all = 0
    for sect_name in include_block:
        group_blocks = getattr(model, sect_name)
        if isinstance(group_blocks, Iterable):
            for (k, block) in enumerate(group_blocks):
                for layer in block.conv:
                    if isinstance(layer, torch.nn.Conv2d):
                        n_weights = layer.weight.numel()
                        layer.weight.copy_(weights[n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
                        n_weights_all += n_weights

def get_weight_block_vec(model, include_block=['down', 'up']):
    ws = []
    for sect_name in include_block:
        group_blocks = getattr(model, sect_name)
        if isinstance(group_blocks, Iterable):
            for (k, block) in enumerate(group_blocks):
                for layer in block.conv:
                    if isinstance(layer, torch.nn.Conv2d):
                        ws.append(layer.weight.flatten())
    return torch.cat(ws)

def get_weight_vec(model, norm_layers):
    ws = []
    for name, param in model.named_parameters():
        name = name.replace("module.", "")
        if 'weight' in name and name not in norm_layers and 'skip_conv' not in name:
            ws.append(param.flatten())
    return torch.cat(ws)

def list_norm_layers(model):
    """ compute list of names of all GroupNorm (or BatchNorm2d) layers in the model """

    norm_layers = []
    for (name, module) in model.named_modules():
        name = name.replace('module.', '')
        if isinstance(module, torch.nn.GroupNorm) or isinstance(module,
                torch.nn.BatchNorm2d):
            norm_layers.append(name + '.weight')
            norm_layers.append(name + '.bias')
    return norm_layers

def gaussian_log_prob(observation, proj_recon, sigma):

    assert observation.shape == proj_recon.shape

    dist = torch.distributions.Normal(loc=proj_recon.flatten(), scale=sigma)

    return dist.log_prob(observation.flatten()).sum()

def tv_loss(x):
    """
    Isotropic TV loss.
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])

def PSNR(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    mse = np.mean((np.asarray(reconstruction) - gt)**2)
    if mse == 0.:
        return float('inf')
    if data_range is not None:
        return 20*np.log10(data_range) - 10*np.log10(mse)
    else:
        data_range = np.max(gt) - np.min(gt)
        return 20*np.log10(data_range) - 10*np.log10(mse)

def SSIM(reconstruction, ground_truth, data_range=None):
    gt = np.asarray(ground_truth)
    if data_range is not None:
        return structural_similarity(reconstruction, gt, data_range=data_range)
    else:
        data_range = np.max(gt) - np.min(gt)
        return structural_similarity(reconstruction, gt, data_range=data_range)

def normalize(x, inplace=False):
    if inplace:
        x -= x.min()
        x /= x.max()
    else:
        x = x - x.min()
        x = x / x.max()
    return x
