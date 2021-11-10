import torch
import numpy as np

def sigmoid_gaussian_flow_log_prob(
    X,
    mu,
    cov = None,
    noise_hess_inv = None,
    eps=1e-7,
    thresh=-7,
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

    X = X.clone().clamp(min=eps, max=1 - eps)
    mu = mu.clone().clamp(min=thresh, max=-thresh)

    # logit function
    Z = torch.log(X) - torch.log(1 - X)
    Z = Z.clone().clamp(min=thresh, max=-thresh)

    if cov is not None and noise_hess_inv is None:
        covariance_matrix = cov
    elif noise_hess_inv is not None and cov is None:
        covariance_matrix = noise_hess_inv
    elif cov is not None and noise_hess_inv is not None:
        covariance_matrix = noise_hess_inv + cov

    # compute gaussian density
    suceed = False
    cnt = 0
    while not suceed:
        try:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
            suceed = True
        except:
            covariance_matrix[np.diag_indices(X.shape[0])] += 1e-6
            cnt += 1
            assert cnt < 100

    log_prob = dist.log_prob(Z)

    # compute log-determinant of sigmoid transformation

    log_det = -torch.log(X) - torch.log(1 - X)

    return (log_prob + log_det.sum()) / X.shape[0]


def sigmoid_gaussian_linearised_log_prob(
    X,
    mu,
    cov = None,
    noise_hess_inv = None,
    eps=1e-7,
    thresh=-15,
    ):
    """
    Pushes predictive Gaussian through sigmoid using local linearisation. Computes log-density in 0-1 space.
    args:
        X: observation in 0-1 space flattened
        mu: mean prediction of Gauussian over activations flattened (in NN output space, pre-sigmoid)
        Cov: covariance matrix of uncertainty over activations
        thresh: maximum output of NN for clipping. Should keep large when using this method as it is more stable than exact method (sigmoid_gaussian_flow_log_prob).
    """

    assert mu.shape == X.shape
    assert len(X.shape) == 1

    X = X.clone().clamp(min=eps, max=1 - eps)
    mu = mu.clone().clamp(min=thresh, max=-thresh)

    sig_mu = torch.sigmoid(mu)
    grad = (sig_mu * (1 - sig_mu)).diag()

    if cov is not None and noise_hess_inv is None:
        covariance_matrix = cov
    elif noise_hess_inv is not None and cov is None:
        covariance_matrix = noise_hess_inv
    elif cov is not None and noise_hess_inv is not None:
        covariance_matrix = noise_hess_inv + cov

    # compute linearised variance in 0-1 space
    covariance_matrix = grad @ covariance_matrix @ grad

    # compute gaussian density
    suceed = False
    cnt = 0
    while not suceed:
        try:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
            suceed = True
        except:
            covariance_matrix[np.diag_indices(X.shape[0])] += 1e-6
            cnt += 1
            assert cnt < 100

    log_prob = dist.log_prob(sig_mu)

    return log_prob / X.shape[0]


def gaussian_log_prob(
    X,
    mu,
    cov = None,
    noise_hess_inv = None
    ):

    assert mu.shape == X.shape

    if cov is not None and noise_hess_inv is None:
        covariance_matrix = cov
    elif noise_hess_inv is not None and cov is None:
        covariance_matrix = noise_hess_inv
    elif cov is not None and noise_hess_inv is not None:
        covariance_matrix = noise_hess_inv + cov

    # compute gaussian density
    suceed = False
    cnt = 0
    while not suceed:
        try:
            dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=covariance_matrix)
            suceed = True
        except:
            covariance_matrix[np.diag_indices(X.shape[0])] += 1e-6
            cnt += 1
            assert cnt < 1000

    log_prob = dist.log_prob(X)

    return log_prob / X.shape[0]

def sigmoid_gaussian_exp(mu, cov, num_samples=100):

    assert cov.shape == (len(mu), len(mu))

    suceed = False
    cnt = 0
    while not suceed:
        try:
            dist = \
                torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,
                    covariance_matrix=cov)
            suceed = True
        except:
            cov[np.diag_indices(mu.shape[0])] += 1e-6
            cnt += 1
            assert cnt < 100
    samples = dist.sample((num_samples, ))

    return torch.sigmoid(samples).mean(dim=0)
