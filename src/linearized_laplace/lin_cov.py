import torch

def est_lin_var(block_priors, Jac, noise_mat_inv, return_numpy=False):

    proj_marginal_prior_term = block_priors.matrix_prior_cov_mul(Jac.cuda()) @ Jac.transpose(1, 0).cuda()
    cov = proj_marginal_prior_term - proj_marginal_prior_term @ torch.inverse(noise_mat_inv.to(block_priors.store_device) + proj_marginal_prior_term) @ proj_marginal_prior_term
    return (cov.diag(), cov) if not return_numpy else (cov.diag().detach().cpu().numpy(), cov.detach().cpu().numpy())
