import torch
import numpy as np

def lin_pred_cov(block_priors, Jac, noise_mat_inv):

    assert noise_mat_inv.diag().min() > 0

    proj_marginal_prior_term =\
        ( block_priors.matrix_prior_cov_mul(Jac).detach() @ Jac.transpose(1, 0) )
    cov = proj_marginal_prior_term \
        - proj_marginal_prior_term @ torch.linalg.solve(noise_mat_inv + proj_marginal_prior_term , proj_marginal_prior_term)

    cov[np.diag_indices(cov.shape[0])] += 1e-6

    assert cov.diag().min() > 0

    return cov.diag(), cov

def submatrix_lin_pred_cov_prior(block_priors, Jac):

    idx_list = block_priors.get_idx_parameters_per_block()
    cov_diag_list = []
    cov_list = []
    for i, idx in enumerate(idx_list): 

        proj_marginal_prior_term = \
            ( block_priors.matrix_prior_cov_mul(Jac[:, idx[-2]:idx[-1]], idx=i) @ Jac[:, idx[-2]:idx[-1]].transpose(1, 0) )
        cov = proj_marginal_prior_term 
        cov[np.diag_indices(cov.shape[0])] += 1e-6

        assert cov.diag().min() > 0

        cov_diag_list.append(cov.diag())
        cov_list.append(cov)
        
    return cov_diag_list, cov_list


def low_rank_GP_lin_model_post_pred_cov(block_priors, Jac_x, Jac_y, noise_model_variance_y):

    Kyy = \
        block_priors.matrix_prior_cov_mul(Jac_y) @ Jac_y.transpose(1, 0)
    Kyy[np.diag_indices(Kyy.shape[0])] += noise_model_variance_y
    Kxx = \
        block_priors.matrix_prior_cov_mul(Jac_x) @ Jac_x.transpose(1, 0)
    Kxy = \
        block_priors.matrix_prior_cov_mul(Jac_x) @ Jac_y.transpose(1, 0)
    Kyx = Kxy.T
    cov = Kxx - Kxy @ torch.linalg.solve(Kyy, Kyx)
    
    cov[np.diag_indices(cov.shape[0])] += 1e-6

    assert cov.diag().min() > 0

    return cov.diag(), cov


def submatrix_low_rank_GP_lin_model_prior_cov(block_priors, Jac_x):

    idx_list = block_priors.get_idx_parameters_per_block()
    Kxx_diag_list = []
    Kxx_list = []
    for i, idx in enumerate(idx_list): 

        Kxx = \
            block_priors.matrix_prior_cov_mul(Jac_x[:, idx[-2]:idx[-1]], idx=i) @ Jac_x[:, idx[-2]:idx[-1]].transpose(1, 0) 

        Kxx[np.diag_indices(Kxx.shape[0])] += 1e-6

        assert Kxx.diag().min() > 0

        Kxx_diag_list.append(Kxx.diag())
        Kxx_list.append(Kxx)
        
    return Kxx_diag_list, Kxx_list
