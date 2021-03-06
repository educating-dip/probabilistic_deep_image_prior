from jax.scipy.stats import norm, multivariate_normal  #, poisson, bernoulli
import jax
from jax.nn import sigmoid
import jax.numpy as jnp
from jax import jit, value_and_grad
from .utils import TV, expected_TV, RadialBasisFuncCov, N_TV_entries

from jax.experimental import optimizers
from jax import value_and_grad
from jax.tree_util import tree_map

@jit
def gaussian_ll(y, mu, log_std):
    std = jnp.exp(log_std)
    z = (y-mu)
    return norm.logpdf(z, loc=0, scale=std)

def return_update(objective, opt_update, get_params):

    obj_grad = jit(value_and_grad(objective))

    @jit
    def update(params, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = obj_grad(params)
        grads_norm = tree_map(lambda x: -x/x.shape[0], grads)
        opt_state = opt_update(0, grads_norm, opt_state)
        new_params = get_params(opt_state)
        return new_params, opt_state, value
    
    return update


def gen_direct_normal_centering_MLE_objective(y, marginal_x_std, projector, noise_std_prop=0.1):
    
    def direct_normal_centering_MLE_objective(params):
        

        noise_std = noise_std_prop * jnp.abs(y).mean()
        x_mean = 0.1307
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)

        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) + gaussian_ll(x, x_mean, jnp.log(marginal_x_std) ).sum(axis=0)
    
    return jit(direct_normal_centering_MLE_objective)



def gen_direct_TV_MLE_objective(y, lambd, projector, noise_std_prop=0.1):
    
    def direct_TV_MLE_objective(params):
        
        noise_std = noise_std_prop * jnp.abs(y).mean()
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)
        
        tv = TV(x.reshape(side_size,side_size))
        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) - lambd * tv / N_TV_entries(side_size)
    
    return jit(direct_TV_MLE_objective)


def gen_direct_predcp_TV_MLE_objective(y, lambd, marginal_x_std, projector, noise_std_prop=0.1):
    # here we optimise x and AR_p simultaneously 
    def direct_TV_MLE_objective(params):
        
        noise_std = noise_std_prop * jnp.abs(y).mean()
        x_mean = 0.1307
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)
        AR_p =  sigmoid(params['AR_p'])
        AR_p = jnp.clip(AR_p, a_min=1e-3, a_max=0.999)
        
        sigma2 = marginal_x_std ** 2
        tv, grad = value_and_grad(expected_TV, argnums=2)(side_size, sigma2, AR_p)

        Cov = RadialBasisFuncCov(side_size, sigma2, AR_p)
        normal_LL = multivariate_normal.logpdf(x, mean=jnp.ones(x.shape)*x_mean, cov=Cov)

        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) - lambd * tv / N_TV_entries(side_size) + jnp.log(jnp.abs(grad.sum())) + normal_LL
    
    return jit(direct_TV_MLE_objective) # 


