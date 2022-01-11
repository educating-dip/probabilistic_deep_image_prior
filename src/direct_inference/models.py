from numpyro.distributions import ImproperUniform, constraints
import numpyro.distributions as dist
import jax.numpy as jnp
import numpyro
from .utils import expected_TV, RadialBasisFuncCov, TV, N_TV_entries, RBF_cholesky

from functools import partial
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_value
import numpyro.distributions as dist
from jax import random
import jax.numpy as jnp
from jax import jit

def draw_samples(model, warmup, n_samples, thinning, num_chains, init_dict={}, max_tree_depth=10):

    nuts_kernel = NUTS(model, forward_mode_differentiation=False, max_tree_depth=max_tree_depth, init_strategy=init_to_value(values=init_dict))
    mcmc = MCMC(nuts_kernel, num_warmup=warmup, num_samples=n_samples, thinning=thinning,
     num_chains=num_chains)
    rng_key = random.PRNGKey(0)

    mcmc.run(
        rng_key,
    )
    mcmc.print_summary()
    return mcmc.get_samples(group_by_chain=False)

def direct_model_normal_centering_prior(y, projector, marginal_x_std=0.3, noise_std_prop=0.1):
    
    obs_dim = projector.shape[0]
    full_dim = projector.shape[1]
    side_size = int(full_dim ** 0.5)
    trainset_mean = 0.1307
    noise_std = noise_std_prop * jnp.abs(y).mean()
    
    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    numpyro.sample("x_prior", dist.Normal(jnp.zeros((full_dim)) + trainset_mean, marginal_x_std*jnp.ones((full_dim))), obs=x)
    
    numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size))

    mu = projector @ x
    err = (y - mu ) / noise_std
    numpyro.sample(name="err", fn=dist.Normal(0, jnp.ones(obs_dim)), obs=err)
    
    
def direct_model_predcp_tv_prior(y, projector, lambd=1e4, marginal_x_std=0.3, noise_std_prop=0.1):
    
    obs_dim = projector.shape[0]
    full_dim = projector.shape[1]
    trainset_mean = 0.1307
    side_size = int(full_dim ** 0.5)
    noise_std = noise_std_prop * jnp.abs(y).mean()
    
    sigma2 = marginal_x_std ** 2
#     
    AR_p = numpyro.sample("AR_p", ImproperUniform(constraints.interval(0.001, 0.999), (), event_shape=(1,)))
    # We scale by the number of pixels to simulate a TV cost applied per pixel
    numpyro.sample("AR_p_prior", dist.Exponential(lambd),
                   obs=expected_TV(side_size, sigma2, AR_p)/N_TV_entries(side_size))
    
    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    Cov = RadialBasisFuncCov(side_size, sigma2, AR_p)  
    numpyro.sample("x_prior", 
                   dist.MultivariateNormal(loc=jnp.zeros((full_dim)) + trainset_mean, covariance_matrix=Cov),
                   obs=x)
    
    numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size))

    mu = projector @ x
    err = (y - mu ) / noise_std
    numpyro.sample(name="err", fn=dist.Normal(0, jnp.ones(obs_dim)), obs=err)


def direct_model_tv_prior(y, projector, lambd=1e4, noise_std_prop=0.1):
    
    obs_dim = projector.shape[0]
    full_dim = projector.shape[1]
    side_size = int(full_dim ** 0.5)
    noise_std = noise_std_prop * jnp.abs(y).mean()

    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    
    # We scale by the number of pixels to simulate a TV cost applied per pixel
    tv = numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size)) 
    numpyro.sample(name="tv_prior", fn=dist.Exponential(lambd), obs=tv)
    
    mu = projector @ x
    err = (y - mu) / noise_std
    numpyro.sample(name="err", fn=dist.Normal(loc=0, scale=jnp.ones(obs_dim)), obs=err)


####### Prior sampling models here


def normal_centering_prior(marginal_x_std=0.3):
    
    full_dim = 784 #projector.shape[1]
    side_size = int(full_dim ** 0.5)
    trainset_mean = 0.1307
    
    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    numpyro.sample("x_prior", dist.Normal(jnp.zeros((full_dim)) + trainset_mean, marginal_x_std*jnp.ones((full_dim))), obs=x)
    
    numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size))
    
    
def predcp_tv_prior(lambd=1e4, marginal_x_std=0.3):
    
    full_dim = 784 #projector.shape[1]
    trainset_mean = 0.1307
    side_size = int(full_dim ** 0.5)

    
    sigma2 = marginal_x_std ** 2

    AR_p = numpyro.sample("AR_p", ImproperUniform(constraints.interval(0.001, 0.999), (), event_shape=(1,)))
    # We scale by the number of pixels to simulate a TV cost applied per pixel
    numpyro.sample("AR_p_prior", dist.Exponential(lambd),
                   obs=expected_TV(side_size, sigma2, AR_p)/N_TV_entries(side_size))
    
    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    Cov = RadialBasisFuncCov(side_size, sigma2, AR_p)  
    numpyro.sample("x_prior", 
                   dist.MultivariateNormal(loc=jnp.zeros((full_dim)) + trainset_mean, covariance_matrix=Cov),
                   obs=x)
    numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size))


def tv_prior(lambd=1e4):
    
    full_dim = 784 #projector.shape[1]
    side_size = int(full_dim ** 0.5)

    x = numpyro.sample("x", ImproperUniform(constraints.interval(0, 1), (), event_shape=(full_dim,)))
    
    # We scale by the number of pixels to simulate a TV cost applied per pixel
    tv = numpyro.deterministic('tv', TV(x.reshape(side_size,side_size)) / N_TV_entries(side_size)) 
    numpyro.sample(name="tv_prior", fn=dist.Exponential(lambd), obs=tv)
