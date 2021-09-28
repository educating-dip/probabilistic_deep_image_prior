import hydra
import numpyro
import numpy as np 
import jax.numpy as jnp
import numpyro.distributions as dist
import matplotlib.pyplot as plt
from numpyro.infer import MCMC, NUTS
from jax import random
from jax.experimental import optimizers
from jax import value_and_grad
from jax.tree_util import tree_map
from jax import jit
from jax.scipy.stats import norm
from omegaconf import DictConfig
from dataset.mnist import simulate
from dataset.utils import load_testset_MNIST_dataset, get_standard_ray_trafos
assert numpyro.__version__.startswith("0.8.0")

def TV(x):
    '''Isotropic TV loss with Jax.'''

    h_tv = jnp.abs(jnp.diff(x, axis=-1, n=1)).sum()
    v_tv = jnp.abs(jnp.diff(x, axis=-2, n=1)).sum()
    return h_tv + v_tv

def direct_model_normal_prior(observation, ray_trafo_mat, sigma):

    obs_dim = ray_trafo_mat.shape[0]
    full_dim = ray_trafo_mat.shape[1]
    x = numpyro.sample('x', dist.Normal(jnp.zeros(full_dim), sigma
                       * jnp.ones(full_dim)))
    mu = ray_trafo_mat @ x
    numpyro.sample(name='obs', fn=dist.Normal(mu, jnp.ones(obs_dim)),
                   obs=observation)


def direct_model_tv_prior(observation, ray_trafo_mat, lambda_hyper):

    obs_dim = ray_trafo_mat.shape[0]
    full_dim = ray_trafo_mat.shape[1]
    x = numpyro.sample('x', dist.Normal(jnp.zeros(full_dim), lambda_hyper
                       * jnp.ones(full_dim)))
    tv = TV(x.reshape(28, 28))
    numpyro.sample(name='tv_prior', fn=dist.Exponential(lambda_hyper), obs=tv)
    mu = ray_trafo_mat @ x
    numpyro.sample(name='obs', fn=dist.Normal(mu, jnp.ones(obs_dim)),
                   obs=observation)


@hydra.main(config_path='../cfgs', config_name='config')
def coordinator(cfg : DictConfig) -> None:

    mnist_loader = load_testset_MNIST_dataset()
    ray_trafos = get_standard_ray_trafos(cfg, return_torch_module=True, return_op_mat=True)
    examples = enumerate(mnist_loader)
    _, (example_image, _) = next(examples)
    observation, filtbackproj, example_image = \
        simulate(example_image, ray_trafos, cfg.noise_specs, return_numpy=True)
    
    N = example_image.shape[0]
    example_image = example_image.flatten()
    observation = observation.flatten()
    M = observation.shape[0]

    kwards = {'ray_trafo_mat': ray_trafos['ray_trafo_mat'].reshape(M, N**2), 
                 'space': ray_trafos['space'], 
                 'tv_scaling': cfg.exp.TV_map.tv_scaling
    }
    params = {'x': filtbackproj.flatten()}

    '''TV MAP subroutines'''

    @jit
    def gaussian_ll(observation, mu, log_std):
        std = jnp.exp(log_std)
        z = (observation-mu)
        return norm.logpdf(z, loc=0, scale=std)

    def gen_direct_TV_MLE_objective(observation, tv_scaling, ray_trafo_mat, space):
        
        def direct_TV_MLE_objective(params):
            x = params['x']
            x_shape = space.shape 
            tv = TV(x.reshape(*x_shape))
            y_pred = ray_trafo_mat @ x
            return gaussian_ll(observation, y_pred, 0).sum(axis=0) - tv_scaling * tv
        
        return jit(direct_TV_MLE_objective)

    @jit
    def update(params, opt_state):
        '''Compute the gradient for a batch and update the parameters'''

        value, grads = obj_grad(params)
        grads_norm = tree_map(lambda x: -x/x.shape[0], grads)
        opt_state = opt_update(0, grads_norm, opt_state)
        return get_params(opt_state), opt_state, value

    objective = gen_direct_TV_MLE_objective(observation, **kwards)
    obj_grad = jit(value_and_grad(objective))
    step_size = 1e-2
    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)

    '''TV Optimisation'''
    for i in range(cfg.exp.TV_map.iters):
        params, opt_state, LL = update(params, opt_state)

    # simple viz. 
    plt.figure(dpi=120)
    plt.imshow(params['x'].reshape(N,N))
    plt.title('MAP with TV regulariser')
    plt.savefig('map_with_TV_reg.png')

    if not cfg.exp.mcmc.run:
        quit()

    nuts_kernel_normal = NUTS(direct_model_normal_prior)
    mcmc = MCMC(nuts_kernel_normal, num_warmup=cfg.exp.mcmc.num_warmup, num_samples=cfg.exp.mcmc.num_samples)
    rng_key = random.PRNGKey(0)

    mcmc.run(
        rng_key,
        observation=observation,
        ray_trafo_mat=kwards['ray_trafo_mat'], 
        sigma=cfg.exp.mcmc.sigma
    )
    # mcmc.print_summary()
    normal_samples = mcmc.get_samples()

    nuts_kernel_tv = NUTS(direct_model_tv_prior)
    mcmc = MCMC(nuts_kernel_tv, num_warmup=cfg.exp.mcmc.num_warmup, num_samples=cfg.exp.mcmc.num_samples)
    rng_key = random.PRNGKey(0)

    mcmc.run(
        rng_key,
        observation=observation,
        ray_trafo_mat=kwards['ray_trafo_mat'], 
        lambda_hyper=cfg.exp.mcmc.lambda_hyper
    )
    # mcmc.print_summary()
    tv_samples = mcmc.get_samples()

    # simple viz. 
    recon_mean_normal = normal_samples['x'].mean(axis=0).reshape(N,N)
    recon_std_normal = normal_samples['x'].std(axis=0).reshape(N,N)
    recon_mean_tv = tv_samples['x'].mean(axis=0).reshape(N,N)
    recon_std_tv = tv_samples['x'].std(axis=0).reshape(N,N)

    _, axs = plt.subplots(2, 2, figsize=(8, 8),  facecolor='w', edgecolor='k', constrained_layout=True, dpi=140)
    axs = axs.flatten()
    axs[0].imshow(recon_mean_normal)
    axs[0].set_title('MCMC mean (normal prior)')
    axs[1].imshow(recon_std_normal)
    axs[1].set_title('mcmc std (normal prior)')
    axs[2].imshow(recon_mean_tv)
    axs[2].set_title('mcmc mean (TV prior)')
    axs[3].imshow(recon_std_tv)
    axs[3].set_title('mcmc std (TV prior)')
    plt.savefig('prior_comparison.pdf') 

    
if __name__ == '__main__':
    coordinator()
