import torch
import xitorch.linalg
import numpy as np
from tqdm import tqdm
from math import ceil
import datetime
import socket
import os
from torch.utils.tensorboard import SummaryWriter
from .sample_based_weights_linearization import weights_linearization, assemble_prior_diag
from scalable_linearised_laplace import (
    get_cov_obs_low_rank, get_cov_obs_low_rank_via_jac_low_rank, fwAD_JvP_batch_ensemble, prior_cov_obs_mat_mul_jac_low_rank,
    vec_weight_prior_cov_mul, vec_jac_mul_batch, generate_closure, generate_low_rank_closure
    )
from .xitorch_solvers import gmres, cg, bicgstab

jitter = 1

def generate_closure_via_low_rank_jac(ray_trafos, bayesianized_model, jac,
        log_noise_model_variance_obs, vec_batch_size, side_length, 
        masked_cov_grads=None, add_noise_model_variance_obs=True):

    def closure(v):
        # takes input (side_length x batchsize)
        v = v.T.view(vec_batch_size, *side_length)
        out = prior_cov_obs_mat_mul_jac_low_rank(ray_trafos, bayesianized_model,
            jac,
            v, log_noise_model_variance_obs,
            masked_cov_grads=masked_cov_grads, 
            add_noise_model_variance_obs=add_noise_model_variance_obs)
        out = out.view(vec_batch_size, np.prod(side_length))
        return out.T
    return closure

def sample_weights_from_diag_prior(bayesianized_model, vec_batch_size, prior_variance_vec, device):

    if len(prior_variance_vec) != 1:
        num_weights_per_prior_var_list = (bayesianized_model.ref_num_params_per_modules_under_gp_priors + 
                                            bayesianized_model.ref_num_params_per_modules_under_normal_priors )
        prior_variance_vec = assemble_prior_diag(
            num_weights_per_prior_var_list,
            prior_variance_vec
            )[None, :]
    else:
        pass
    
    return (prior_variance_vec**.5 ) * torch.randn(
                (vec_batch_size, bayesianized_model.num_params_under_all_priors),
                device=device
            )

class xitorch_operator(xitorch.LinearOperator):
    def __init__(self, size, func, device='cpu', dtype=torch.float64):
        super().__init__(shape=(size,size), is_hermitian=True, device=device, dtype=dtype)
        self.func = func

    def _getparamnames(self, prefix=""):
       return []
    
    def _mv(self, v):
        return self.func(v.T).T

def samples_from_posterior_using_CG( 
        cfg, 
        ray_trafos, filtbackproj, observation,
        model_batched, bayesianized_model, be_model, be_modules,
        log_noise_model_variance_obs,
        prior_variance_vec,
        preconditioner,
        device=None, 
        jac=None
    ):

    if device is None:
        device = bayesianized_model.store_device

    num_batches = ceil(
            cfg.sample_based_mrglik.EM_cycle.mc_samples / cfg.sample_based_mrglik.impl.vec_batch_size
            )
    samples_obs_variance = torch.zeros(
        observation.numel(), device=device
        )
    samples_weights_variance = torch.zeros(
        bayesianized_model.num_params_under_all_priors, device=device
        )

    res_norm_list = []
    for _ in tqdm(range(num_batches),
        desc='sample_observation_from_posterior_using_CG', miniters=num_batches//100):

        sample_weight_vec = sample_weights_from_diag_prior(
            bayesianized_model, 
            cfg.sample_based_mrglik.impl.vec_batch_size, 
            prior_variance_vec, 
            device
        ) # \theta^{'} (batch_size, params)

        s_image = fwAD_JvP_batch_ensemble(
            filtbackproj,
            be_model,
            sample_weight_vec,
            be_modules
        ) # J\theta^{'}

        s_image = s_image.squeeze(dim=1) # remove trivial sample-batch dimension (be_model uses B_ensemble x B_sample x C x H x W)
        s_observation = ray_trafos['ray_trafo_module'](s_image) # AJ\theta^{'}
        noise_term = ( (log_noise_model_variance_obs.exp() + jitter )**.5) * torch.randn(
                            (cfg.sample_based_mrglik.impl.vec_batch_size, observation.numel()),
                            device=device
                        )
        obs = noise_term - s_observation.view(cfg.sample_based_mrglik.impl.vec_batch_size, -1) # batch_size, observation.numel()
        obs_norms = torch.norm(obs, 2, dim=-1, keepdim=True)  
        obs_scaled = obs.div(obs_norms)

        if jac is not None:
            operator_closure = generate_closure_via_low_rank_jac(ray_trafos, bayesianized_model, jac,
                log_noise_model_variance_obs, cfg.sample_based_mrglik.impl.vec_batch_size, observation.shape[1:], 
                masked_cov_grads=None, add_noise_model_variance_obs=True)
        else:
            operator_closure = generate_closure(
                ray_trafos, filtbackproj, 
                bayesianized_model, model_batched, be_model, be_modules,
                torch.log(log_noise_model_variance_obs.exp() + jitter),
                cfg.sample_based_mrglik.impl.vec_batch_size, 
                side_length=observation.shape[1:],
                masked_cov_grads=None,
                use_fwAD_for_jvp=True,
                add_noise_model_variance_obs=True
            )

        main_closure = xitorch_operator(
            np.prod(observation.shape[1:]),
            func = operator_closure,
            device=device,
            dtype=torch.float64
        )

        if preconditioner is not None: 
            precon_closure = generate_low_rank_closure(
                preconditioner
            )
            precon = xitorch_operator(
                        np.prod(observation.shape[1:]),
                        func=precon_closure,
                        device=device, 
                        dtype=torch.float64
                    )
        else: 
            precon = None
     
        max_cg_iter = cfg.sample_based_mrglik.linear_cg.max_cg_iter
        tol = cfg.sample_based_mrglik.linear_cg.tolerance
        if cfg.sample_based_mrglik.linear_cg.method == 'cg': 
            solve_T, residual_norm = cg(
                main_closure, obs_scaled.T,
                posdef=True,
                precond=precon, 
                max_niter=max_cg_iter, rtol=tol, atol=1e-08, eps=1e-6, resid_calc_every=10, verbose=True)
        elif cfg.sample_based_mrglik.linear_cg.method == 'bicgstab':
            solve_T, residual_norm = bicgstab(
                main_closure, obs_scaled.T,
                posdef=True,
                precond_l=precon,
                max_niter=max_cg_iter, rtol=tol, atol=1e-08, eps=1e-6, resid_calc_every=10, verbose=True)
        elif cfg.sample_based_mrglik.linear_cg.method == 'gmres':
            solve_T, residual_norm = gmres(
                main_closure, obs_scaled.T,
                posdef=True,
                max_niter=max_cg_iter, rtol=tol, atol=1e-08, eps=1e-6)
        else: 
            raise NotImplementedError
        
        res_norm_list.append(
            residual_norm.mean().item()
            )
        
        inv_obs_diff = solve_T.T * obs_norms
        inv_image = ray_trafos['ray_trafo_module_adj']( # A^T Kyy^{-1}AJ\theta^{'}
            inv_obs_diff.view(*s_observation.shape)
            )

        delta_params = vec_weight_prior_cov_mul( # Σ_θ J^T A^T Kyy^{-1} A J \th1eta^{'}
            bayesianized_model, 
            vec_jac_mul_batch( # J^TA^TKyy^{-1}AJ\theta^{'}
                    model_batched,
                    filtbackproj,
                    inv_image.view(cfg.sample_based_mrglik.impl.vec_batch_size, -1), 
                    bayesianized_model
                )
        )
        
        sample_weights_from_posterior = (sample_weight_vec + delta_params
            ).detach() # (\theta^{'} - Σ_θ J^T A^T Kyy^{-1} A J \theta^{'})
        
        sample_obs_from_posterior = ray_trafos['ray_trafo_module']( # A J * (\theta^{'} - Σ_θ J^T A^T Kyy^{-1} A J \theta^{'})
            fwAD_JvP_batch_ensemble(
                filtbackproj, 
                be_model, 
                sample_weights_from_posterior,
                be_modules
            ).view(*s_image.shape)
        ).detach().to(device)
    
        samples_weights_variance += sample_weights_from_posterior.detach().pow(2).sum(dim=0).flatten()
        samples_obs_variance += sample_obs_from_posterior.detach().pow(2).sum(dim=0).flatten()

    return (samples_obs_variance / (cfg.sample_based_mrglik.impl.vec_batch_size*num_batches), 
                samples_weights_variance / (cfg.sample_based_mrglik.impl.vec_batch_size*num_batches), 
                res_norm_list
                )

def prior_E_step_update(
    k,
    cfg,
    bayesianized_model, model, modules, model_batched, be_model, be_modules,
    filtbackproj, observation, example_image, 
    ray_trafos, san_ray_trafos, 
    preconditioner,
    prior_variance_vec,
    log_noise_model_variance_obs,
    initial_map_weights, 
    jac
    ):

    linearized_weights, lin_pred, mse_loss = weights_linearization(
            cfg=cfg, 
            bayesianized_model=bayesianized_model, model=model, all_modules_under_prior=modules,
            filtbackproj=filtbackproj, observation=observation, ground_truth=example_image,
            ray_trafos=ray_trafos, 
            prior_variance_vec=prior_variance_vec, 
            noise_model_variance_obs=log_noise_model_variance_obs.exp(), 
            initial_map_weights=initial_map_weights
        )

    if cfg.sample_based_mrglik.impl.save_linearized_weights:

        torch.save(
            {'linearized_weights_step_in_EM_cycle': linearized_weights,
                'linearized_prediction': lin_pred},  
                './linearized_weights_{}.pt'.format(k)
        )
    
    # sample obs using CG sampling
    samples_obs_variance, sample_weights_variance, res_norm_list = samples_from_posterior_using_CG(
        cfg,
        san_ray_trafos, filtbackproj, observation,
        model_batched, bayesianized_model, be_model, be_modules,
        log_noise_model_variance_obs,
        prior_variance_vec,
        preconditioner,
        device=None, 
        jac=jac
        )

    return samples_obs_variance, sample_weights_variance, res_norm_list, linearized_weights, mse_loss

def prior_M_step_update(
    samples_obs_variance,
    samples_weights_variance,
    prior_variance_vec, 
    noise_model_variance_obs, 
    linearized_weights, 
    mse_loss,
    num_observation, 
    num_weights_per_prior_var_list, 
    min_value_noise_variance=1e-4,
    min_value_params_variance=1e-5
    ):
    
    with torch.no_grad():

        eff_dim_obs = (
            samples_obs_variance * (noise_model_variance_obs**-1)
                ).clamp(max=1.).sum()
        
        noise_model_variance_obs = mse_loss / (num_observation - eff_dim_obs)

        if len(prior_variance_vec) == 1: 
            prior_variance_vec = ( linearized_weights.pow(2).sum(dim=0) / eff_dim_obs) * torch.ones(1, device=samples_obs_variance.device)
        else:
            idx = 0
            new_prior_prec_list = []
            eff_dim_weights = 0
            assert len(prior_variance_vec) == len(num_weights_per_prior_var_list)
            for _, (prior_var_i, num_weights_i) in enumerate(
                    zip(prior_variance_vec, num_weights_per_prior_var_list)):
                eff_dim_weights_i = ( num_weights_i - (prior_var_i**(-1) * samples_weights_variance[idx:idx+num_weights_i]
                        ).clamp(max=1).sum(dim=0) )
                new_prior_prec_i = eff_dim_weights_i/linearized_weights[idx:idx+num_weights_i].pow(2).sum(dim=0).clamp(min=1e-6)
                new_prior_prec_list.append(new_prior_prec_i)
                eff_dim_weights += eff_dim_weights_i
                idx += num_weights_i
            assert eff_dim_weights > 0 
            prior_variance_vec = (torch.stack(new_prior_prec_list, dim=0) * eff_dim_obs / eff_dim_weights)**-1

    return (prior_variance_vec.clamp(min=min_value_params_variance),
                noise_model_variance_obs.clamp(min=min_value_noise_variance).log(),
                eff_dim_obs
            )

def set_priors_variances(bayesianized_model, prior_variance_vec):

    parameters = (bayesianized_model.gp_log_variances + bayesianized_model.normal_log_variances)
    if len(prior_variance_vec) != 1:
        for param, prior_variance in zip(parameters, prior_variance_vec):
            param.data = torch.log(prior_variance)
    else:
        for param in parameters:
            param.data = torch.log(prior_variance_vec)

def sample_based_EM_hyperparams_optim(

        cfg, 
        ray_trafos, san_ray_trafos,
        filtbackproj, observation, example_image,
        bayesianized_model, model, modules, model_batched, be_model, be_modules, jac
        ):
    
    if san_ray_trafos is None: 
        san_ray_trafos = ray_trafos

    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    comment = 'sample_based_mrglik_opt'
    logdir = os.path.join(
        './', comment + '_' +  current_time + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=logdir)
    
    log_noise_model_variance_obs = torch.log(
        torch.tensor(cfg.sample_based_mrglik.EM_cycle.noise_model_variance_obs,
        device=bayesianized_model.store_device)
        )

    # construct preconditioner for CG sampling
    preconditioner = None
    if cfg.sample_based_mrglik.impl.use_preconditioner: 
        if cfg.sample_based_mrglik.preconditioner.preconditioner_path is None: 
            reduced_rank_dim = cfg.sample_based_mrglik.preconditioner.reduced_rank_dim + cfg.sample_based_mrglik.preconditioner.oversampling_param
            max_rank_dim = np.prod(ray_trafos['ray_trafo'].range.shape)
            if reduced_rank_dim > max_rank_dim:
                reduced_rank_dim = max_rank_dim
                print(
                    'low_rank preconditioner: rank changed to full rank ({:d})'.format(reduced_rank_dim))
            random_matrix_T = torch.randn(
                    (reduced_rank_dim, *ray_trafos['ray_trafo'].range.shape),
                    device=bayesianized_model.store_device
                    )
            if jac is None: 
                U, L, inv_cov_obs_approx = get_cov_obs_low_rank(random_matrix_T, san_ray_trafos, filtbackproj.to(bayesianized_model.store_device),
                    bayesianized_model, model_batched, be_model, be_modules,
                    torch.log(log_noise_model_variance_obs.exp() + jitter),
                    cfg.sample_based_mrglik.impl.vec_batch_size, 
                    reduced_rank_dim - cfg.sample_based_mrglik.preconditioner.oversampling_param, 
                    cfg.sample_based_mrglik.preconditioner.oversampling_param, use_fwAD_for_jvp=True)
            else:
                U, L, inv_cov_obs_approx = get_cov_obs_low_rank_via_jac_low_rank(random_matrix_T, ray_trafos, filtbackproj.to(bayesianized_model.store_device), bayesianized_model, 
                    jac,
                    torch.log(log_noise_model_variance_obs.exp() + jitter),
                    cfg.sample_based_mrglik.impl.vec_batch_size,
                    reduced_rank_dim - cfg.sample_based_mrglik.preconditioner.oversampling_param, 
                    cfg.sample_based_mrglik.preconditioner.oversampling_param
                    )

            preconditioner = U, L, inv_cov_obs_approx

            if cfg.sample_based_mrglik.preconditioner.save_preconditioner: 
                torch.save(
                    preconditioner, './sampled_based_mrglik_preconditioner.pt'
                )
        else:
            preconditioner = torch.load(os.path.join(
                    cfg.sample_based_mrglik.preconditioner.preconditioner_path, 'sampled_based_mrglik_preconditioner.pt'),
                    map_location=bayesianized_model.store_device
                )
    
    prior_variance_vec = cfg.sample_based_mrglik.priors.variance_init * torch.ones(1,
        device=bayesianized_model.store_device
        )

    linearized_weights = None
    for k in tqdm(
        range(cfg.sample_based_mrglik.EM_cycle.num_cycles),
            desc='sample_based_EM_hyperparams_optim'):

        writer.add_scalar('noise_model_variance_obs', log_noise_model_variance_obs.exp().item(), k)

        for i, gp_log_variance in enumerate(bayesianized_model.gp_log_variances):
            writer.add_scalar('gp_variance_{}'.format(i),
                    torch.exp(gp_log_variance).item(), k)
        for i, normal_log_variance in enumerate(bayesianized_model.normal_log_variances):
            writer.add_scalar('normal_variance_{}'.format(i),
                            torch.exp(normal_log_variance).item(), k)

        if k == cfg.sample_based_mrglik.EM_cycle.tied_variance_steps:
            prior_variance_vec = prior_variance_vec * torch.ones(
                    len(bayesianized_model.ref_num_params_per_modules_under_gp_priors + bayesianized_model.ref_num_params_per_modules_under_normal_priors), 
                    device=bayesianized_model.store_device
                )

        samples_obs_variance, samples_weights_variance, res_norm_list, linearized_weights, mse_loss = prior_E_step_update(
            k=k,
            cfg=cfg,
            bayesianized_model=bayesianized_model, model=model, modules=modules, model_batched=model_batched, be_model=be_model, be_modules=be_modules,
            filtbackproj=filtbackproj.to(bayesianized_model.store_device), 
            observation=observation.to(bayesianized_model.store_device), 
            example_image=example_image.to(bayesianized_model.store_device), 
            ray_trafos=ray_trafos, san_ray_trafos=san_ray_trafos,
            preconditioner=preconditioner, 
            prior_variance_vec=prior_variance_vec,
            log_noise_model_variance_obs=log_noise_model_variance_obs,
            initial_map_weights=linearized_weights,
            jac=jac
            )

        prior_variance_vec, log_noise_model_variance_obs, eff_dim_obs = prior_M_step_update(
            samples_obs_variance=samples_obs_variance,
            samples_weights_variance=samples_weights_variance,
            noise_model_variance_obs=log_noise_model_variance_obs.exp(),
            prior_variance_vec=prior_variance_vec,
            linearized_weights=linearized_weights,
            mse_loss=mse_loss,
            num_observation=observation.numel(),
            num_weights_per_prior_var_list=bayesianized_model.ref_num_params_per_modules_under_gp_priors + bayesianized_model.ref_num_params_per_modules_under_normal_priors, 
        )
        
        writer.add_scalar('eff_dim_obs', eff_dim_obs.item(), k)
        writer.add_scalar('norm_w_map', linearized_weights.pow(2).sum().item(), k)
        writer.add_scalar('mean residual norm CG', np.mean(res_norm_list), k)
        
        set_priors_variances(bayesianized_model, prior_variance_vec)

        if k == 1: 
            cfg.sample_based_mrglik.weights_linearization.iterations = int( cfg.sample_based_mrglik.weights_linearization.iterations * 0.75)
            cfg.sample_based_mrglik.weights_linearization.lr = float(cfg.sample_based_mrglik.weights_linearization.lr * 0.5)
        
        if k <= cfg.sample_based_mrglik.EM_cycle.fixed_noise_variance_steps:
            log_noise_model_variance_obs = torch.log(
                torch.tensor(
                    cfg.sample_based_mrglik.EM_cycle.noise_model_variance_obs,
                    device=bayesianized_model.store_device
                    )
            )
