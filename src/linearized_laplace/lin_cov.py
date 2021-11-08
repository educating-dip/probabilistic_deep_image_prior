import torch
import numpy as np

def compute_lin_pred_cov(block_priors, Jac, noise_mat_inv):

    assert noise_mat_inv.diag().min() > 0
    
    proj_marginal_prior_term = \
        ( block_priors.matrix_prior_cov_mul(Jac).detach() @ Jac.transpose(1,
            0) )
    cov = proj_marginal_prior_term - proj_marginal_prior_term @ torch.linalg.solve(noise_mat_inv + proj_marginal_prior_term , proj_marginal_prior_term)

    cov[np.diag_indices(cov.shape[0])] += 1e-6

    assert cov.diag().min() > 0

    return cov.diag(), cov

def compute_submatrix_lin_pred_cov(block_priors, Jac):

    idx_list = block_priors.get_idx_parameters_per_block()
    cov_diag_list = []
    cov_list = []
    for i, idx in enumerate(idx_list): 

        proj_marginal_prior_term = \
            ( block_priors.matrix_prior_cov_mul(Jac[:, idx[-2]:idx[-1]], idx=i) @ Jac[:, idx[-2]:idx[-1]].transpose(1,
                0) )
        cov = proj_marginal_prior_term 
        cov[np.diag_indices(cov.shape[0])] += 1e-6

        assert cov.diag().min() > 0

        cov_diag_list.append(cov.diag())
        cov_list.append(cov)
        
    return cov_diag_list, cov_list