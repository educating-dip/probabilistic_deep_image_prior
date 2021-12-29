import torch 
from gpytorch.utils.lanczos import lanczos_tridiag_to_diag
from gpytorch.utils import StochasticLQ, linear_cg
from .prior_cov_obs import prior_cov_obs_mat_mul
from .log_det_grad import compose_masked_cov_grad_from_modules

def generate_probes(side_length, num_random_probes, dtype=None, device=None, jacobi_vector=None):
    probe_vectors = torch.empty(side_length, num_random_probes, dtype=dtype, device=device)
    probe_vectors.bernoulli_().mul_(2).add_(-1)
    if jacobi_vector is not None:
        assert len(jacobi_vector.shape) == 1
        probe_vectors *= jacobi_vector.pow(0.5).unsqueeze(1)
    return probe_vectors  # side_length, num_random_probes

def generate_closure(ray_trafos, filtbackproj, bayesianized_model, hooked_model, 
        be_model, be_modules, log_noise_model_variance_obs, vec_batch_size, 
        masked_cov_grad_tuple=None, use_fwAD_for_jvp=True, add_noise_model_variance_obs=True):

    def closure(v):
        v = v.T.view(vec_batch_size, 1, 10, 41)
        out = prior_cov_obs_mat_mul(ray_trafos, filtbackproj, bayesianized_model, hooked_model, 
            be_model, be_modules, v, log_noise_model_variance_obs, masked_cov_grad_tuple=masked_cov_grad_tuple,
            use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=add_noise_model_variance_obs)
        out = out.view(vec_batch_size, 410)
        return out.T
    return closure

def stochastic_LQ_logdet_grad(closure, grad_closure, probe_vectors, side_length, vec_batch_size, max_cg_iter, tolerance, jacobi_vector=None):

    num_random_probes = probe_vectors.shape[1]
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
    # estimate gradient
    grad = (grad_closure(solves) * conditioned_probes * probe_vector_norms.pow(2)).sum(dim=0).mean(dim=0) # we re-introduce the norms to make sure probes are K=I

    return grad, logdet_term + logdet_correction, solves

def compute_approx_log_det_grad(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, use_fwAD_for_jvp=True, jacobi_vector=None):
    
    grads = {}

    # v * (AJSigma_thetaJ.TA.T + sigma^2_y)
    main_closure = generate_closure(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, masked_cov_grad_tuple=None, use_fwAD_for_jvp=use_fwAD_for_jvp, add_noise_model_variance_obs=True)
    probe_vectors = generate_probes(side_length=410, num_random_probes=vec_batch_size, device=bayesianized_model.store_device, jacobi_vector=jacobi_vector) 
    gp_priors_grad_dict, normal_priors_grad_dict, _ = compose_masked_cov_grad_from_modules(bayesianized_model, log_noise_model_variance_obs)
    
    for gp_prior in bayesianized_model.gp_priors:
        # building AJ(delta Sigma_theta/delta_hyperparams)J^T A^T
        grad_closure = generate_closure(ray_trafos, filtbackproj, bayesianized_model, hooked_model, fwAD_be_model, fwAD_be_modules, log_noise_model_variance_obs, vec_batch_size, masked_cov_grad_tuple=(gp_priors_grad_dict['lengthscales'][gp_prior], normal_priors_grad_dict['all_zero']), use_fwAD_for_jvp=True, add_noise_model_variance_obs=False)
        grad, _, _  = stochastic_LQ_logdet_grad(main_closure, grad_closure, probe_vectors, side_length=410, vec_batch_size=vec_batch_size, max_cg_iter=10, tolerance=1e-1, jacobi_vector=jacobi_vector)
        grads[gp_prior.cov.log_lengthscale] = grad
    
    return grads

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