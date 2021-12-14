import torch
from opt_einsum import contract

def _get_cov_from_modules(bayesianized_model):

    cov_under_gp_prior = []
    for num_filters, prior in zip(bayesianized_model.ref_num_filters_per_modules_under_gp_priors, bayesianized_model.gp_priors):
        for _ in range(num_filters):
            cov_under_gp_prior.append(prior.cov.cov_mat(return_cholesky=False))
    cov_under_normal_prior = []
    for num_params, prior in zip(bayesianized_model.ref_num_filters_per_modules_under_normal_priors, bayesianized_model.normal_priors): 
        for _ in range(num_params): 
            cov_under_normal_prior.append(torch.exp(prior.log_variance))
    return torch.stack(cov_under_gp_prior), torch.stack(cov_under_normal_prior)

def _fast_prior_cov_mul(sliced_cov_mat, v):
    
    v = v.view(-1, sliced_cov_mat.shape[0], sliced_cov_mat.shape[-1])
    v = v.permute(1, 0, 2)
    vCov_mul = contract('nxk,nkc->ncx', v, sliced_cov_mat)
    vCov_mul = vCov_mul.reshape([sliced_cov_mat.shape[0]* sliced_cov_mat.shape[-1], 1]).t()
    return vCov_mul

def matrix_with_prior_cov(bayesianized_model, v):

    cov_under_gp_prior, cov_under_normal_prior = _get_cov_from_modules(bayesianized_model)
    cov_under_gp_prior = _fast_prior_cov_mul(cov_under_gp_prior, v)
    pass