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
        if 'AR_p' in params.keys():
            new_params['AR_p'] = new_params['AR_p'].clip(a_min=0, a_max=1-1e-5)
        return new_params, opt_state, value
    
    return update


def gen_direct_normal_centering_MLE_objective(y, marginal_x_std, projector):
    
    def direct_normal_centering_MLE_objective(params):
        
        noise_std = 0.1
        x_mean = 0.1307
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)

        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) + gaussian_ll(x, x_mean, jnp.log(marginal_x_std) ).sum(axis=0)
    
    return jit(direct_normal_centering_MLE_objective)



def gen_direct_TV_MLE_objective(y, lambd, projector):
    
    def direct_TV_MLE_objective(params):
        
        noise_std = 0.1
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)
        
        tv = TV(x.reshape(side_size,side_size))
        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) - lambd * tv / N_TV_entries(side_size)
    
    return jit(direct_TV_MLE_objective)


def gen_direct_predcp_TV_MLE_objective(y, lambd, marginal_x_std, projector):
    # here we optimise x and AR_p simultaneously 
    def direct_TV_MLE_objective(params):
        
        noise_std = 0.1
        x_mean = 0.1307
        
        x = sigmoid(params['x']) # constrain to [0, 1]
        side_size = int(x.shape[0] ** 0.5)
        AR_p = params['AR_p'] # preconstrained
        
        sigma2 = marginal_x_std ** 2
        tv = expected_TV(side_size, sigma2, AR_p).sum()
        
        Cov = RadialBasisFuncCov(side_size, sigma2, AR_p)
        normal_LL = multivariate_normal.logpdf(x, mean=jnp.ones(x.shape)*x_mean, cov=Cov)
        
        y_pred = projector @ x
        return gaussian_ll(y, y_pred, jnp.log(noise_std)).sum(axis=0) - lambd * tv / N_TV_entries(side_size) + normal_LL
    
    return jit(direct_TV_MLE_objective) # 


