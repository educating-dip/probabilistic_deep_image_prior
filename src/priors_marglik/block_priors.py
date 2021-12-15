import torch
import torch.nn as nn
import numpy as np
from .bayes_blocks import get_GPprior, BayesianiseBlock, BayesianiseBlockUp
from .bayes_layer import Conv2dGPprior
from .priors import NormalPrior
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from deep_image_prior import tv_loss

def _cov_mat_mul(x, cov_mat):
    N = x.shape[0]
    x = x.view(-1, cov_mat.shape[0], cov_mat.shape[-1])
    x = x.permute(1, 0, 2)
    out = x @ cov_mat
    out = out.permute(0, 2, 1).reshape([cov_mat.shape[0]
            * cov_mat.shape[-1], N]).t()
    return out

class BlocksGPpriors(nn.Module):

    def __init__(self, model, bayesianize_model, store_device, lengthscale_init, variance_init, lin_weights=None):
        super(BlocksGPpriors, self).__init__()

        self.model = model
        self.bayesianize_model = bayesianize_model
        self.store_device = store_device
        self.lengthscale_init = lengthscale_init
        self.variance_init = variance_init
        # self.priors = torch.nn.ModuleList(self._assemble_block_priors())
        # self.gp_priors = torch.nn.ModuleList(self._assemble_gp_priors())
        # self.normal_priors = torch.nn.ModuleList(self._assemble_normal_priors())
        self.gp_priors = bayesianize_model.gp_priors
        self.normal_priors = bayesianize_model.normal_priors
        self.lin_weights = lin_weights
        # self.num_params = \
        #     len([param for param in self.priors.parameters() if param.requires_grad]) // 2

    # def _assemble_block_priors(self):
    #     model_sections = ['down', 'up']
    #     mirror_blcks = []
    #     for sect_name in model_sections:
    #         group_blocks = getattr(self.model, sect_name)
    #         if isinstance(group_blocks, Iterable):
    #             for block in group_blocks:
    #                 mirror_blcks.append((BayesianiseBlock(block,
    #                                     self.store_device,
    #                                     self.lengthscale_init,
    #                                     self.variance_init) if sect_name
    #                                     == 'down'
    #                                      else BayesianiseBlockUp(block,
    #                                     self.store_device,
    #                                     self.lengthscale_init,
    #                                     self.variance_init)))
    #     return mirror_blcks

    # def _assemble_gp_priors(self):
    #     gp_priors = []
    #     for gp_prior in self.bayesianize_model.gp_priors:
    #         gp_priors.append(get_GPprior(
    #                 store_device=self.store_device,
    #                 lengthscale_init=self.lengthscale_init,
    #                 variance_init=self.variance_init))
    #     return gp_priors

    # def _assemble_normal_priors(self):
    #     normal_priors = []
    #     for normal_prior in self.bayesianize_model.normal_priors:
    #         normal_priors.append(NormalPrior(kernel_size=1,
    #             variance_init = self.variance_init,
    #             store_device = self.store_device))
    #     return normal_priors

    # @property
    # def log_lengthscales(self):
    #     return [param for (name, param) in self.priors.named_parameters()
    #             if param.requires_grad and name.find('log_lengthscale')
    #             != -1]

    @property
    def gp_log_lengthscales(self):
        return [gp_prior.cov.log_lengthscale for gp_prior in self.gp_priors
                if gp_prior.cov.log_lengthscale.requires_grad]

    # @property
    # def log_variances(self):
    #     return [param for (name, param) in self.priors.named_parameters()
    #             if param.requires_grad and name.find('log_variance') != -1]

    @property
    def gp_log_variances(self):
        return [gp_prior.cov.log_variance for gp_prior in self.gp_priors
                if gp_prior.cov.log_variance.requires_grad]

    @property
    def normal_log_variances(self):
        return [normal_prior.log_variance for normal_prior in self.normal_priors
                if normal_prior.log_variance.requires_grad]

    # def _get_repeat(self, module):

    #     repeat = 0
    #     for el in module.block.conv:
    #         if isinstance(el, Conv2dGPprior):
    #             repeat += el.in_channels * el.out_channels
    #     return repeat

    def _get_repeat(self, modules):

        repeat = 0
        for module in modules:
            repeat += module.in_channels * module.out_channels
        return repeat

    # def get_idx_parameters_per_block(self):

    #     model_sections = ['down', 'up']
    #     n_weights_all = 0
    #     list_idx = []
    #     for sect_name in model_sections:
    #         group_blocks = getattr(self.model, sect_name)
    #         if isinstance(group_blocks, Iterable):
    #             for _, block in enumerate(group_blocks): 
    #                 n_weights_per_block = 0
    #                 for layer in block.conv:
    #                     if isinstance(layer, torch.nn.Conv2d):
    #                         params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
    #                         n_weights_per_block += params.numel()
    #                 list_idx.append((n_weights_all, n_weights_all + n_weights_per_block))
    #                 n_weights_all += n_weights_per_block
    #     return list_idx

    def get_idx_parameters_per_block(self):

        n_weights_all = 0
        list_idx = []
        for _, modules_under_prior in zip(
                self.priors, self.bayesianize_model.ref_modules_under_priors):
            n_weights_per_block = 0
            for layer in modules_under_prior:
                assert isinstance(layer, torch.nn.Conv2d)
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                n_weights_per_block += params.numel()
            list_idx.append((n_weights_all, n_weights_all + n_weights_per_block))
            n_weights_all += n_weights_per_block
        return list_idx

    # def get_net_log_det_cov_mat(self, ):

    #     log_det = torch.zeros(1, device=self.store_device)
    #     for prior in self.priors:
    #         log_det = log_det + prior.GPp.cov.log_det() * self._get_repeat(prior)
    #     return log_det

    def get_net_log_det_cov_mat(self):

        log_det = torch.zeros(1, device=self.store_device)
        for gp_prior, modules_under_gp_prior in zip(
                self.gp_priors, self.bayesianize_model.ref_modules_under_gp_priors):
            log_det = log_det + gp_prior.cov.log_det() * self._get_repeat(modules_under_gp_prior)
        for normal_prior, modules_under_normal_prior in zip(
                self.normal_priors, self.bayesianize_model.ref_modules_under_normal_priors):
            log_det = log_det + normal_prior.cov_log_det() * self._get_repeat(modules_under_normal_prior)
        return log_det

    # def get_net_prior_log_prob(self, ):

    #     model_sections = ['down', 'up']
    #     log_prob = torch.zeros(1, device=self.store_device)
    #     i = 0 
    #     for sect_name in model_sections:
    #         group_blocks = getattr(self.model, sect_name)
    #         if isinstance(group_blocks, Iterable):
    #             for _, block in enumerate(group_blocks):
    #                 for layer in block.conv:
    #                     if isinstance(layer, torch.nn.Conv2d):
    #                         params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
    #                         log_prob += self.priors[i].GPp.log_prob(params).sum(dim=0)
    #                 i += 1
    #     return log_prob

    def get_net_prior_log_prob(self):

        log_prob = torch.zeros(1, device=self.store_device)
        for gp_prior, modules_under_gp_prior in zip(
                self.gp_priors, self.bayesianize_model.ref_modules_under_gp_priors):
            for layer in modules_under_gp_prior:
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                log_prob += gp_prior.log_prob(params).sum(dim=0)
        for normal_prior, modules_under_normal_prior in zip(
                self.normal_priors, self.bayesianize_model.ref_modules_under_normal_priors):
            for layer in modules_under_normal_prior:
                params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
                log_prob += normal_prior.log_prob(params).sum(dim=0)
        return log_prob

    # def get_net_prior_log_prob_lin_weights(self, lin_weights):

    #     model_sections = ['down', 'up']
    #     n_weights_all = 0
    #     i = 0
    #     log_prob = torch.zeros(1, device=self.store_device)
    #     for sect_name in model_sections:
    #         group_blocks = getattr(self.model, sect_name)
    #         if isinstance(group_blocks, Iterable):
    #             for _, block in enumerate(group_blocks):
    #                 for layer in block.conv:
    #                     if isinstance(layer, torch.nn.Conv2d):
    #                         params = layer.weight.view(-1, *layer.kernel_size).view(-1, layer.kernel_size[0]**2)
    #                         lin_weights = self.lin_weights[n_weights_all:n_weights_all + params.numel()]
    #                         assert params.flatten().shape == lin_weights.shape
    #                         lin_weights = lin_weights.view_as(params)
    #                         n_weights_all += params.numel()
    #                         log_prob += self.priors[i].GPp.log_prob(lin_weights).sum(dim=0)
    #                 i += 1 
    #     return log_prob

    @property
    def priors(self):
        return chain(self.gp_priors, self.normal_priors)

    def _gp_prior_cov_mats(self, gp_prior, modules_under_gp_prior):
        cov_mat = gp_prior.cov.cov_mat(return_cholesky=False)
        repeat_fct = self._get_repeat(modules_under_gp_prior)
        return [cov_mat] * repeat_fct

    def _normal_prior_cov_mats(self, normal_prior, modules_under_normal_prior):
        cov_mat = normal_prior.cov_mat(return_cholesky=False)
        repeat_fct = self._get_repeat(modules_under_normal_prior)
        return [cov_mat] * repeat_fct

    def get_gp_prior_cov_mat(self, gp_idx=None):

        gp_priors = self.gp_priors
        ref_modules_under_gp_priors = self.bayesianize_model.ref_modules_under_gp_priors
        if gp_idx is not None:
            gp_priors = gp_priors[gp_idx]
            ref_modules_under_gp_priors = ref_modules_under_gp_priors[gp_idx]
            if not isinstance(gp_priors, Iterable):
                gp_priors = [gp_priors]
                ref_modules_under_gp_priors = [ref_modules_under_gp_priors]

        gp_cov_mat_list = []
        for gp_prior, modules_under_gp_prior in zip(
                gp_priors, ref_modules_under_gp_priors):
            gp_cov_mat_list += self._gp_prior_cov_mats(gp_prior, modules_under_gp_prior)
        gp_cov_mat = torch.stack(gp_cov_mat_list)
        return gp_cov_mat

    # normal_idx is relative, i.e. an index in the self.normal_priors list
    def get_normal_prior_cov_mat(self, normal_idx=None):

        normal_priors = self.normal_priors
        ref_modules_under_normal_priors = self.bayesianize_model.ref_modules_under_normal_priors
        if normal_idx is not None:
            normal_priors = normal_priors[normal_idx]
            ref_modules_under_normal_priors = ref_modules_under_normal_priors[normal_idx]
            if not isinstance(normal_priors, Iterable):
                normal_priors = [normal_priors]
                ref_modules_under_normal_priors = [ref_modules_under_normal_priors]

        normal_cov_mat_list = []
        for normal_prior, modules_under_normal_prior in zip(
                normal_priors, ref_modules_under_normal_priors):
            normal_cov_mat_list += self._normal_prior_cov_mats(normal_prior, modules_under_normal_prior)
        normal_cov_mat = torch.stack(normal_cov_mat_list) if normal_cov_mat_list else torch.empty(0, 1)
        return normal_cov_mat

    def get_single_prior_cov_mat(self, idx):
        if idx < len(self.gp_priors):
            gp_idx = idx
            cov_mat = self.get_gp_prior_cov_mat(gp_idx=gp_idx)
        else:
            normal_idx = idx - len(self.gp_priors)
            cov_mat = self.get_normal_prior_cov_mat(normal_idx=normal_idx)
        return cov_mat

    def matrix_prior_cov_mul(self, x, idx=None):

        if idx is None:

            gp_cov_mat = self.get_gp_prior_cov_mat()
            normal_cov_mat = self.get_normal_prior_cov_mat()

            gp_x = x[:, :(gp_cov_mat.shape[0] * gp_cov_mat.shape[-1])]
            normal_x = x[:, (gp_cov_mat.shape[0] * gp_cov_mat.shape[-1]):]

            gp_out = _cov_mat_mul(gp_x, gp_cov_mat)
            if normal_x.shape[1] != 0:
                normal_out = _cov_mat_mul(normal_x, normal_cov_mat)
            else:
                normal_out = torch.empty(x.shape[0], 0).to(x.device)

            out = torch.cat([gp_out, normal_out], dim=-1)

        elif np.isscalar(idx):

            cov_mat = self.get_single_prior_cov_mat(idx=idx)
            out = _cov_mat_mul(x, cov_mat)

        else:
            raise NotImplementedError

        return out
