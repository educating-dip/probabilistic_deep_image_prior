import torch
import torch.nn as nn
import numpy as np
import itertools
from torch import linalg
from torch.distributions.multivariate_normal import MultivariateNormal

class RadialBasisFuncCov(nn.Module):

    def __init__(
        self,
        kernel_size,
        lengthscale_init,
        variance_init,
        dist_func,
        ):

        super(RadialBasisFuncCov, self).__init__()
        self.kernel_size = kernel_size
        self.dist_mat = self.compute_dist_matrix(dist_func)
        self.log_lengthscale = nn.Parameter(torch.Tensor(1))
        self.log_variance = nn.Parameter(torch.Tensor(1))
        self._init_parameters(lengthscale_init, variance_init)

    def _init_parameters(self, lengthscale_init, variance_init):
        nn.init.constant_(self.log_lengthscale,
                          np.log(lengthscale_init))
        nn.init.constant_(self.log_variance, np.log(variance_init))

    def cov_mat(self, return_cholesky=True):
        lengthscale = torch.exp(self.log_lengthscale) + 1e-6
        variance = torch.exp(self.log_variance) + 1e-6
        cov_mat = variance * torch.exp(-self.dist_mat / lengthscale)
        return (linalg.cholesky(cov_mat) if return_cholesky else cov_mat)

    def compute_dist_matrix(self, dist_func):
        coords = [torch.as_tensor([i, j], dtype=torch.float32) for i in
                  range(self.kernel_size) for j in
                  range(self.kernel_size)]
        combs = [[el_1, el_2] for el_1 in coords for el_2 in coords]
        dist_mat = torch.as_tensor([dist_func(el1 - el2) for (el1,
                                   el2) in combs], dtype=torch.float32)
        return dist_mat.view(self.kernel_size ** 2, self.kernel_size
                             ** 2)

class GPprior(nn.Module):

    def __init__(self, covariance_function):
        super(GPprior, self).__init__()
        self.cov = covariance_function

    def sample(self, shape):

        cov = self.cov.cov_mat(return_cholesky=True)
        mean = torch.zeros(self.cov.kernel_size ** 2)
        m = MultivariateNormal(loc=mean, scale_tril=cov)
        params_shape = shape + [self.cov.kernel_size,
                                self.cov.kernel_size]
        return m.rsample(sample_shape=shape).view(params_shape)

if __name__ == '__main__':

    dist_func = lambda x: linalg.norm(x, ord=2)
    cov_kwards = {
        'kernel_size': 3,
        'lengthscale_init': 1,
        'variance_init': 1,
        'dist_func': dist_func,
        }
    cov_func = RadialBasisFuncCov(**cov_kwards)
    p = GPprior(cov_func)
    samples = p.sample(shape=[32, 3])
    print(samples.size())
