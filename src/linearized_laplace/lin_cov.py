import torch

def est_lin_var(block_priors, Jac, noise_mat_inv):

    proj_marginal_prior_term = \
        ( block_priors.matrix_prior_cov_mul(Jac).detach() @ Jac.transpose(1,
            0) ).cpu()
    cov = proj_marginal_prior_term - proj_marginal_prior_term @ torch.linalg.solve(noise_mat_inv + proj_marginal_prior_term, proj_marginal_prior_term)

    return cov.diag(), cov
