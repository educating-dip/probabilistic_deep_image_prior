import torch
import numpy as np
from copy import deepcopy

#!/usr/bin/python
# -*- coding: utf-8 -*-


def sigmoid_guassian_log_prob(
    X,
    mu,
    Cov,
    eps=1e-7,
    thresh=-5,
    ):
    """
    Computes log-density of observations under Gaussian transofrmed
    by sigmoid. Uses change of variables formula.
    args:
        X: observation in 0-1 space
        mu: mean prediction of Gauussian over activations
        Cov: covariance matrix of uncertainty over activations
    """

    assert mu.shape == X.shape
    assert Cov.shape == (len(X), len(X))

    X = X.clone().clamp(min=eps, max=1 - eps)
    mu = mu.clone().clamp(min=thresh, max=-thresh)

    # logit function
    Z = torch.log(X) - torch.log(1 - X)
    Z = Z.clone().clamp(min=thresh, max=-thresh)

    # compute gaussian density
    suceed = False
    while not suceed:
        try:
            dist = \
                torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,
                    covariance_matrix=Cov)
            suceed = True
        except:
            Cov[np.diag_indices(X.shape[0])] += 1e-4
    log_prob = dist.log_prob(Z)

    # compute log-determinant of sigmoid transformation

    log_det = -torch.log(X) - torch.log(1 - X)

    return (log_prob + log_det.sum()) / X.shape[0]


def sigmoid_gaussian_exp(mu, Cov, num_samples=100):

    assert Cov.shape == (len(mu), len(mu))

    suceed = False
    while not suceed:
        try:
            dist = \
                torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,
                    covariance_matrix=Cov)
            suceed = True
        except:
            Cov[np.diag_indices(mu.shape[0])] += 1e-4
    samples = dist.sample((num_samples, ))

    return torch.sigmoid(samples).mean(dim=0)
