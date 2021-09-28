import torch
import numpy as np
from skimage.metrics import structural_similarity

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
