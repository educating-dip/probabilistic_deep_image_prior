import torch 
import numpy as np
from gpytorch.utils.lanczos import lanczos_tridiag_to_diag
from gpytorch.utils import StochasticLQ, linear_cg
from .prior_cov_obs import prior_cov_obs_mat_mul
from .log_det_grad import compose_masked_cov_grad_from_modules
from .vec_weight_prior_mul_closure import vec_weight_prior_cov_mul_base
from .batch_jac import vec_op_jac_mul_batch

def generate_probes(side_length, num_random_probes, dtype=None, device=None, jacobi_vector=None):
    probe_vectors = torch.empty(side_length, num_random_probes, dtype=dtype, device=device)
    probe_vectors.bernoulli_().mul_(2).add_(-1)
    if jacobi_vector is not None:
        assert len(jacobi_vector.shape) == 1
        probe_vectors *= jacobi_vector.pow(0.5).unsqueeze(1)
    return probe_vectors  # side_length, num_random_probes


def generate_closure(ray_trafos, filtbackproj, bayesianized_model, hooked_model, 
        be_model, be_modules, log_noise_model_variance_obs, vec_batch_size, side_length, 
        masked_cov_grads=None, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True):

    def closure(v):
        # takes input (side_length x batchsize)
        v = v.T.view(vec_batch_size, *side_length)
        out = prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, 
            be_model, be_modules, v, log_noise_model_variance_obs, masked_cov_grads=masked_cov_grads,
            use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=add_noise_model_variance_obs)
        out = out.view(vec_batch_size, np.prod(side_length))
        return out.T
    return closure


def stochastic_LQ_logdet_and_solves(closure, probe_vectors, max_cg_iter, tolerance, jacobi_vector=None):

    num_random_probes = probe_vectors.shape[1]
    side_length = probe_vectors.shape[0]
    probe_vector_norms = torch.norm(probe_vectors, 2, dim=-2, keepdim=True)  # 1, num_random_probes; for rademacher random variates the norm is equal to sqrt(side_length)
    probe_vectors_scaled = probe_vectors.div(probe_vector_norms) # side_length, num_random_probes
    
    if jacobi_vector is not None:
        preconditioning_closure = generate_jacobi_closure(jacobi_vector)
        logdet_correction = jacobi_vector.log().sum()
        conditioned_probes = preconditioning_closure(probe_vectors_scaled)
    else:
        preconditioning_closure = None
        logdet_correction = 0
        conditioned_probes = probe_vectors_scaled

    solves, tmat = linear_cg(closure, probe_vectors_scaled, n_tridiag=num_random_probes, tolerance=tolerance,
                        eps=1e-10, stop_updating_after=1e-10, max_iter=max_cg_iter,
                        max_tridiag_iter=max_cg_iter-1, preconditioner=preconditioning_closure)
    
    # estimate log-determinant
    slq = StochasticLQ(max_iter=-1, num_random_probes=num_random_probes)
    pos_eigvals, pos_eigvecs = lanczos_tridiag_to_diag(tmat)
    (logdet_term,) = slq.evaluate((side_length, side_length), pos_eigvals, pos_eigvecs, [lambda x: x.log()])
    
    conditioned_scaled_probes = conditioned_probes * probe_vector_norms.pow(2)  # we re-introduce the norms to make sure probes are K=I
    return solves, conditioned_scaled_probes, logdet_term + logdet_correction 


def compute_approx_log_det_grad(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, side_length, use_fwAD_for_jvp=True, jacobi_vector=None):
    
    grads = {}

    # v * (A * J * Σ_θ * J.T * A.T + σ^2_y)
    # we need to detach log_noise_model_variance_obs
    main_closure = generate_closure(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs.detach(), vec_batch_size, side_length=side_length, masked_cov_grads=None, use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=True)
    probe_vectors = generate_probes(side_length=np.prod(side_length), num_random_probes=vec_batch_size, device=bayesianized_model.store_device, jacobi_vector=jacobi_vector) 
    gp_priors_grad_dict, normal_priors_grad_dict, log_noise_variance_obs_grad_dict = compose_masked_cov_grad_from_modules(bayesianized_model, log_noise_model_variance_obs.detach())

    solves, cs_probes, log_det_term  = stochastic_LQ_logdet_and_solves(main_closure, probe_vectors, max_cg_iter=50, tolerance=1, jacobi_vector=jacobi_vector)

    solves_reshape = solves.T.view(vec_batch_size, *side_length)
    solves_AJ_reshape = vec_op_jac_mul_batch(ray_trafos, hooked_model, filtbackproj, solves_reshape, bayesianized_model)
    solves_AJ = solves_AJ_reshape

    cs_probes_reshape = cs_probes.T.view(vec_batch_size, *side_length)
    cs_probes_AJ_reshape = vec_op_jac_mul_batch(ray_trafos, hooked_model, filtbackproj, cs_probes_reshape, bayesianized_model)
    cs_probes_AJ = cs_probes_AJ_reshape

    for gp_prior in bayesianized_model.gp_priors:
        for param_name in ['lengthscales', 'variances']: 
            solves_AJSig = vec_weight_prior_cov_mul_base(bayesianized_model, gp_priors_grad_dict[param_name][gp_prior], normal_priors_grad_dict['all_zero'], solves_AJ)
            grad = (solves_AJSig * cs_probes_AJ).sum(dim=1, keepdim=True).mean(dim=0).detach()
            if param_name == 'lengthscales': 
                grads[gp_prior.cov.log_lengthscale] = 0.5 * grad # added 0.5 minimize
            elif param_name == 'variances': 
                grads[gp_prior.cov.log_variance] = 0.5 * grad
    
    for normal_prior in bayesianized_model.normal_priors:

        solves_AJSig = vec_weight_prior_cov_mul_base(bayesianized_model, gp_priors_grad_dict['all_zero'], normal_priors_grad_dict['variances'][normal_prior], solves_AJ)
        grad = (solves_AJSig * cs_probes_AJ).sum(dim=1, keepdim=True).mean(dim=0).detach()
        grads[normal_prior.log_variance] = 0.5 * grad

    solves_AJSig = vec_weight_prior_cov_mul_base(bayesianized_model, log_noise_variance_obs_grad_dict['gp_prior'], log_noise_variance_obs_grad_dict['normal_prior'], solves_AJ)    
    grads[log_noise_model_variance_obs] = 0.5 * (solves_AJSig * cs_probes_AJ).sum(dim=1, keepdim=True).mean(dim=0).detach()
    
    return grads, log_det_term

def generate_jacobi_closure(jacobi_vec, eps=1e-3):
    assert len(jacobi_vec.shape) == 1
    mat_ = jacobi_vec.clone().clamp(min=eps).pow(-1)
    def closure(v):
        assert v.shape[0] == mat_.shape[0]
        if len(v.shape) == 1:
            return v * mat_
        elif len(v.shape) == 2:
            return v * mat_.unsqueeze(1)
        else:
            raise NotImplementedError
    return closure