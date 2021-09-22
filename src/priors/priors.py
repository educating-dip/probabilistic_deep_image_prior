import torch
import torch.nn as nn
import itertools
from torch.distributions.multivariate_normal import MultivariateNormal


class RadialBasisFuncCov(nn.Module):

    def __init__(
        self,
        kernel_size,
        lengthscale_init,
        precision_init,
        dist_func,
        ):

        super(RadialBasisFuncCov, self).__init__()
        self.dist_mat = self.compute_dist_matrix(kernel_size, dist_func)
        self.lengthscale = nn.Parameter(torch.Tensor(1))
        self.precision = nn.Parameter(torch.Tensor(1))
        self._init_parameters(lengthscale_init, precision_init)

    def _init_parameters(self, lengthscale_init, precision_init):
        nn.init.constant_(self.lengthscale, lengthscale_init)
        nn.init.constant_(self.precision, precision_init)

    def cov_mat(self):
        lengthscale = torch.exp(self.lengthscale) + 1e-6
        precision = torch.exp(self.precision) + 1e-6
        return precision * torch.exp(-self.dist_mat / lengthscale)

    def compute_dist_matrix(self, kernel_size, dist_func):
        coords = [torch.as_tensor([i, j], dtype=torch.float32) for i in
                  range(kernel_size) for j in range(kernel_size)]
        combs = [[el_1, el_2] for el_1 in coords for el_2 in coords]
        dist_mat = torch.as_tensor([dist_func(el1 - el2) for (el1,
                                   el2) in combs], dtype=torch.float32)
        return dist_mat.view(kernel_size ** 2, kernel_size ** 2)


class MultiVariateGaussianPrior(nn.Module):

    def __init__(self, mean, covariance_function):
        super(MultiVariateGaussianPrior, self).__init__()
        self.cov = covariance_function
        self.mean = mean

    def sample(self, shape):
        m = MultivariateNormal(loc=self.mean,
                               covariance_matrix=self.cov.cov_mat())
        return m.rsample(sample_shape=shape)


if __name__ == '__main__':

    dist_func = lambda x: torch.linalg.norm(x, ord=2)
    cov_kwards = {
        'kernel_size': 3,
        'lengthscale_init': 2,
        'precision_init': 2,
        'dist_func': dist_func,
        }
    cov_func = RadialBasisFuncCov(**cov_kwards)
    mean = torch.zeros((9, 9))
    p = MultiVariateGaussianPrior(mean, cov_func)
    samples = p.sample(shape=(32, 3))
    print(samples.size())
