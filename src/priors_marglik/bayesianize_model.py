import torch
import torch.nn as nn
import torch.linalg as linalg
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain
from .priors import GPprior, RadialBasisFuncCov, NormalPrior

class BayesianizeModel(nn.Module):

    def __init__(
        self,
        reconstructor,
        lengthscale_init,
        variance_init,
        include_normal_priors=True,
        ):
        
        super().__init__()
        self.store_device = reconstructor.device
        self.include_normal_priors = include_normal_priors
        self.gp_priors = nn.ModuleList([])
        self.normal_priors = nn.ModuleList([])
        self.ref_modules_under_gp_priors = []
        self.ref_modules_under_normal_priors = []
        self.init_priors(
                reconstructor, 
                {'lengthscale_init': lengthscale_init,
                    'variance_init': variance_init
                }
            )

        self.ref_num_filters_per_modules_under_gp_priors = self._get_num_filters_under_priors(
                self.ref_modules_under_gp_priors
            )
        
        self.ref_num_params_per_modules_under_normal_priors = self._get_num_filters_under_priors(
                self.ref_modules_under_normal_priors
            )

    
    def _extract_blocks_from_model(self, model):
        return [block for block in model.children()]
    
    def _extract_Conv2d_modules(self, sub_block):
        return [module for module in sub_block.modules() if isinstance(module, torch.nn.Conv2d)]
    
    def _extract_kernel_size_Conv2d(self, modules):
        kernel_size_list = []
        for module in modules: 
            kernel_size_list.append(module.kernel_size[-1])
        return kernel_size_list
    
    def _find_modules_under_gp_prior(self, modules, kernel_size=3):

        kernel_size_list = self._extract_kernel_size_Conv2d(modules)
        if kernel_size in kernel_size_list: 
            if all(el == kernel_size_list[0] for el in kernel_size_list):
                return modules
            else:
                return modules[:-1]
        else:
            return []
    
    def _find_modules_under_normal_prior(self, modules, kernel_size=1):

        kernel_size_list = self._extract_kernel_size_Conv2d(modules)
        if kernel_size in kernel_size_list: 
            if all(el == kernel_size_list[0] for el in kernel_size_list):
                return modules
            else: 
                return [modules[kernel_size_list.index(kernel_size)]]
        else:
            return []
    
    def _remove_modules_from_inactive_skip_ch(self, modules, sub_block):

        if hasattr(sub_block, 'skip'):
            if not sub_block.skip:
                return modules[:-1]
            else:
                return modules
        else:
            return modules

    def init_priors(self, reconstructor, priors_kwards):

        def _add_priors_from_modules(modules, priors_kwards): 

            modules_gp_priors = self._find_modules_under_gp_prior(modules)
            modules_normal_priors = self._find_modules_under_normal_prior(modules) if self.include_normal_priors else []
            if modules_gp_priors: self._add_gp_priors(modules_gp_priors, **priors_kwards)
            if modules_normal_priors: self._add_normal_priors(modules_normal_priors, 
                **{'variance_init': priors_kwards['variance_init']})
         
        blocks = self._extract_blocks_from_model(reconstructor.model)
        for block in blocks:
            if isinstance(block, Iterable):
                for sub_block in block:
                    modules = self._extract_Conv2d_modules(sub_block)
                    modules = self._remove_modules_from_inactive_skip_ch(modules, sub_block)
                    _add_priors_from_modules(modules, priors_kwards)
            else:
                modules = self._extract_Conv2d_modules(block)
                _add_priors_from_modules(modules, priors_kwards)

    def _add_gp_priors(self, modules, lengthscale_init, variance_init):

        dist_func = lambda x: linalg.norm(x, ord=2)
        cov_kwards = {
            'kernel_size': 3,
            'lengthscale_init': lengthscale_init,
            'variance_init': variance_init,
            'dist_func': dist_func,
            'store_device': self.store_device,
            }
        cov_func = \
            RadialBasisFuncCov(**cov_kwards).to(self.store_device)
        GPp = GPprior(cov_func, self.store_device)
        self.gp_priors.append(GPp)
        self.ref_modules_under_gp_priors.append(modules)

    def _add_normal_priors(self, modules, variance_init):

        normal_prior = NormalPrior(kernel_size=1,
                variance_init = variance_init,
                store_device = self.store_device)
        self.normal_priors.append(normal_prior)
        self.ref_modules_under_normal_priors.append(modules)

    @property
    def priors(self):
        return chain(self.gp_priors, self.normal_priors)

    @property
    def ref_modules_under_priors(self):
        return self.ref_modules_under_gp_priors + self.ref_modules_under_normal_priors
    
    @property
    def num_params_under_priors(self): 
        return sum(self.ref_num_filters_per_modules_under_gp_priors) * 3**2 + \
                sum(self.ref_num_params_per_modules_under_normal_priors)

    def get_all_modules_under_prior(self):
        all_modules = []
        for modules in self.ref_modules_under_gp_priors:
            all_modules += modules
        for modules in self.ref_modules_under_normal_priors:
            all_modules += modules
        return all_modules
    
    def _get_num_filters_under_priors(self, modules_under_priors):

        num_filters_under_priors = []
        for modules in modules_under_priors:
            num_filters_per_modules = 0
            for module in modules:
                num_filters_per_modules +=  module.in_channels * module.out_channels
            num_filters_under_priors.append(num_filters_per_modules)
        return num_filters_under_priors
