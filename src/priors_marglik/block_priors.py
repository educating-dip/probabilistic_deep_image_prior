import torch
import torch.nn as nn
from .bayes_blocks import BayesianiseBlock, BayesianiseBlockUp
from .bayes_layer import Conv2dGPprior
from collections.abc import Iterable
from copy import deepcopy
from deep_image_prior import tv_loss

def require_grad_false(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

class BlocksGPpriors(nn.Module):

    def __init__(self, model, store_device, lengthscale_init):
        super(BlocksGPpriors, self).__init__()

        self.model = require_grad_false(model)
        self.store_device = store_device
        self.lengthscale_init = lengthscale_init
        self.num_mc_samples = 5
        self.priors = torch.nn.ModuleList(self._assemble_block_priors())
        self.num_params = \
            len([param for param in self.priors.parameters() if param.requires_grad]) // 2

    def _assemble_block_priors(self):
        model_sections = ['down', 'up']
        mirror_blcks = []
        for sect_name in model_sections:
            group_blocks = getattr(self.model, sect_name)
            if isinstance(group_blocks, Iterable):
                for block in group_blocks:
                    mirror_blcks.append((BayesianiseBlock(block,
                                        self.store_device,
                                        self.lengthscale_init) if sect_name
                                        == 'down'
                                         else BayesianiseBlockUp(block,
                                        self.store_device,
                                        self.lengthscale_init)))
        return mirror_blcks

    @property
    def log_lengthscales(self):
        return [param for (name, param) in self.priors.named_parameters()
                if param.requires_grad and name.find('log_lengthscale')
                != -1]


    @property
    def log_variances(self):
        return [param for (name, param) in self.priors.named_parameters()
                if param.requires_grad and name.find('log_variance') != -1]

    def get_expected_TV_loss(self, x):

        model_sections = ['down', 'up']
        exp_tv_loss = []
        i = 0
        for sect_name in model_sections:
            group_blocks = getattr(self.model, sect_name)
            if isinstance(group_blocks, Iterable):
                for k, block in enumerate(group_blocks):
                    if sect_name  == 'down':
                        original_block = deepcopy(self.model.down[k])
                        self.model.down[k] = self.priors[i]
                    elif sect_name == 'up':
                        original_block = deepcopy(self.model.up[k])
                        self.model.up[k] = self.priors[i]
                    tv_loss_samples = torch.zeros(1, device=self.store_device)
                    for _ in range(self.num_mc_samples):
                        recon = self.model.forward(x)[0]
                        tv_loss_samples = tv_loss_samples + tv_loss(recon)
                    exp_tv_loss.append(tv_loss_samples/self.num_mc_samples)
                    # reverting back to the original model
                    if sect_name == 'down':
                        self.model.down[k] = original_block
                    elif sect_name == 'up':
                        self.model.up[k] = original_block
                    i += 1
        return exp_tv_loss

    def _get_repeat(self, module):

        repeat = 0
        for el in module.block.conv:
            if isinstance(el, Conv2dGPprior):
                repeat += el.in_channels * el.out_channels
        return repeat


    def get_net_log_det_cov_mat(self, ):

        log_det = torch.zeros(1, device=self.store_device)
        for prior in self.priors:
            log_det = log_det + prior.GPp.cov.log_det() * self._get_repeat(prior)
        return log_det

    def get_net_prior_log_prob(self, ):

        model_sections = ['down', 'up']
        mirror_blcks = []
        log_prob = torch.zeros(1, device=self.store_device)
        for sect_name in model_sections:
            group_blocks = getattr(self.model, sect_name)
            if isinstance(group_blocks, Iterable):
                for k, block in enumerate(group_blocks):
                    for layer in block.conv:
                        if isinstance(layer, torch.nn.Conv2d):
                            params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                            log_prob += self.priors[k].GPp.log_prob(params).sum(dim=0)
        return log_prob

    def get_net_prior_cov_mat(self, ):

        cov_blocks = []
        for prior in self.priors:
            cov_mat = prior.GPp.cov.cov_mat()
            repeat_fct = self._get_repeat(prior)
            for _ in range(repeat_fct):
                cov_blocks.append(cov_mat)
        return torch.stack(cov_blocks)

    def matrix_prior_cov_mul(self, x):

        N = x.shape[0]
        tensor_cov_mat = self.get_net_prior_cov_mat()
        x = x.view(-1, tensor_cov_mat.shape[0], tensor_cov_mat.shape[-1])
        x = x.permute(1, 0, 2)
        out = x @ tensor_cov_mat
        out = out.permute(0, 2, 1).reshape([tensor_cov_mat.shape[0]
                * tensor_cov_mat.shape[-1], N]).t()
        return out