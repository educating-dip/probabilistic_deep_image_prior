import torch
from opt_einsum import contract

def _compose_cov_from_modules(bayesianized_model):

    cov_under_gp_prior = []
    for num_filters, prior in zip(
            bayesianized_model.ref_num_filters_per_modules_under_gp_priors,
                bayesianized_model.gp_priors):
        cov_under_gp_prior.append(
                prior.cov.cov_mat(return_cholesky=False).repeat(num_filters, 1, 1)
            )

    cov_under_normal_prior = []
    for num_params, prior in zip(
            bayesianized_model.ref_num_params_per_modules_under_normal_priors, 
                bayesianized_model.normal_priors): 
        cov_under_normal_prior.append(
                torch.exp(prior.log_variance).repeat(num_params, 1, 1)
                )
    
    return torch.cat(cov_under_gp_prior), torch.cat(cov_under_normal_prior)

def _fast_prior_cov_mul(v, sliced_cov_mat):
    
    N = v.shape[0]
    v = v.view(-1, sliced_cov_mat.shape[0], sliced_cov_mat.shape[-1])
    v = v.permute(1, 0, 2)
    vCov_mul = contract('nxk,nkc->ncx', v, sliced_cov_mat)
    vCov_mul = vCov_mul.reshape([sliced_cov_mat.shape[0]* sliced_cov_mat.shape[-1], N]).t()
    return vCov_mul

def matrix_with_prior_cov(bayesianized_model, v):

    cov_under_gp_prior, cov_under_normal_prior = _compose_cov_from_modules(bayesianized_model)
    gp_v = v[:, :(cov_under_gp_prior.shape[0] * cov_under_gp_prior.shape[-1])]
    normal_v = v[:, (cov_under_gp_prior.shape[0] * cov_under_gp_prior.shape[-1]):]
    v_mul_cov_under_gp_prior = _fast_prior_cov_mul(gp_v, cov_under_gp_prior)
    v_mul_cov_under_normal_prior = _fast_prior_cov_mul(normal_v, cov_under_normal_prior)
    vmulSigma_params = torch.cat([v_mul_cov_under_gp_prior, v_mul_cov_under_normal_prior], dim=-1)
    return vmulSigma_params 
