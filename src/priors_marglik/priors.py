import torch
import torch.nn as nn
import numpy as np
import itertools
from torch import linalg
try:
    from torch.linalg import cholesky
except:
    from torch import cholesky
from torch.distributions.multivariate_normal import MultivariateNormal

class RadialBasisFuncCov(nn.Module):

    def __init__(
        self,
        kernel_size,
        lengthscale_init,
        variance_init,
        dist_func,
        store_device
        ):

        super(RadialBasisFuncCov, self).__init__()

        self.kernel_size = kernel_size
        self.store_device = store_device
        self.dist_mat = self.compute_dist_matrix(dist_func)
        self.log_lengthscale = nn.Parameter(torch.ones(1, device=self.store_device))
        self.log_variance = nn.Parameter(torch.ones(1, device=self.store_device))
        self._init_parameters(lengthscale_init, variance_init)

    def _init_parameters(self, lengthscale_init, variance_init):
        nn.init.constant_(self.log_lengthscale,
                          np.log(lengthscale_init))
        nn.init.constant_(self.log_variance, np.log(variance_init))
        # self.log_variance.requires_grad=False

    def cov_mat(self, return_cholesky=True, eps=1e-6):

        lengthscale = torch.exp(self.log_lengthscale)
        variance = torch.exp(self.log_variance)
        cov_mat = variance * torch.exp(-self.dist_mat / lengthscale)
        cov_mat[np.diag_indices(cov_mat.shape[0])] += eps
        return (cholesky(cov_mat) if return_cholesky else cov_mat)

    def compute_dist_matrix(self, dist_func):
        coords = [torch.as_tensor([i, j], dtype=torch.float32) for i in
                  range(self.kernel_size) for j in
                  range(self.kernel_size)]
        combs = [[el_1, el_2] for el_1 in coords for el_2 in coords]
        dist_mat = torch.as_tensor([dist_func(el1 - el2) for (el1,
                                   el2) in combs], dtype=torch.float32, device=self.store_device)
        return dist_mat.view(self.kernel_size ** 2, self.kernel_size
                             ** 2)
    def log_det(self):
        return 2 * self.cov_mat(return_cholesky=True).diag().log().sum()

class GPprior(nn.Module):

    def __init__(self, covariance_function, store_device):
        super(GPprior, self).__init__()
        self.cov = covariance_function
        self.store_device = store_device

    def sample(self, shape):
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.store_device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        params_shape = shape + [self.cov.kernel_size,
                                self.cov.kernel_size]
        return m.rsample(sample_shape=shape).view(params_shape)

    def log_prob(self, x):
        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2).to(self.store_device)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        return m.log_prob(x)

if __name__ == '__main__':

    dist_func = lambda x: linalg.norm(x, ord=2)
    cov_kwards = {
        'kernel_size': 3,
        'lengthscale_init': 1,
        'variance_init': 1,
        'dist_func': dist_func,
        'store_device': None
        }
    cov_func = RadialBasisFuncCov(**cov_kwards)
    p = GPprior(cov_func)
    samples = p.sample(shape=[32, 3])
    print(samples.size())
