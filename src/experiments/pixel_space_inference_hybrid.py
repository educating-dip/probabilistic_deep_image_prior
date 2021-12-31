# %%
import os
import sys
N_up = 1
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up] )
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    print(nb_dir)

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import jax
import jax.numpy as jnp
import numpy as np
from direct_inference.utils import *
from direct_inference.models import *
from direct_inference.objectives import *

import pickle
from sklearn.neighbors import KernelDensity

# module load cuda/11.4
# module load gcc/8

# %% Problem setup


img_side = 28
num_angles = 5
noise_std = 0.1

op_mat = gen_op_mat(img_side, num_angles)


kmnist_path = '/scratch4/ja666/dip_bayesian_ext/kmnist'

train_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=True))
test_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=False))


# %% Settings

warmup = 10000
n_samples = 10000
thinning = 2
num_chains = 5

N_images_val = 10
N_images_test = 20

# prior_std_list = [1e-2, 0.05, 1e-1, 0.5, 1, 5, 10]
# TV_lambda_list = [10, 50, 100, 500, 1000, 5000, 10000]
# bandwidth_list = [1e-2, 0.05, 1e-1, 0.5, 1]

N_optimisation_restarts = 5
optimisation_steps = int(1e5)
optimisation_stepsize = 1e-2
optimisation_stop_length = 1000


# %% logging structures

savedir = '../save/Hybrid_HMC/'
os.makedirs(savedir, exist_ok=True)

hyperparam_search_result_dict = {}
bw_search_result_dict = {}
test_result_dict = {}


# %% Choose method to run

sampling_model = direct_model_predcp_tv_prior
optimisation_objective_gen = gen_direct_predcp_TV_MLE_objective

best_std = 5
best_bw = 0.1
best_lambd = 5000


# %% Draw test samples

test_psnr_list = []
test_LL_list = []

test_opt_psnr_mat = np.zeros((N_images_val, N_optimisation_restarts)) * np.nan

goodstart_opt_psnr_list = []

for n_im in range(N_images_val):

    # draw image
    (example_image, _) = next(test_dset)
    image_np = example_image.numpy().flatten()
    observation = op_mat @ image_np
    observation += np.random.randn(*observation.shape) * noise_std * jnp.abs(observation).mean()
    
    # draw samples
    model = partial(sampling_model, observation, op_mat, best_lambd, best_std)
    samples = draw_samples(model, warmup, n_samples, thinning, num_chains)

    reconstruction_mean = samples['x'].mean(axis=0)
    xmean_psnr = psnr(reconstruction_mean, image_np, smax=1)
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bw).fit(samples['x']) 
    LL = kde.score_samples(image_np[None,:]) / len(image_np)

    test_psnr_list.append(xmean_psnr)
    test_LL_list.append(LL)
    
    test_result_dict[f'im_{n_im}_samples'] = samples
    test_result_dict[f'im_{n_im}_xmean_psnr'] = xmean_psnr
    test_result_dict[f'im_{n_im}_LL'] = LL

    # Optimise from random init

    key = random.PRNGKey(0)
    for n_opt in range(N_optimisation_restarts):
        key = random.split(key)[0]

        params = {}
        params['x'] = jax.scipy.special.logit((jax.random.normal(key, shape=(784,)) * 0.35 + 0.1307 ).clip(a_min=1e-3, a_max=1-1e-3))  # sample from marginal distribution of data roughly
        params['AR_p'] = jnp.zeros(1) * 0

        objective = optimisation_objective_gen(observation, best_lambd, best_std, op_mat)

        opt_init, opt_update, get_params = optimizers.adam(optimisation_stepsize) #optimizers.momentum(step_size, mass=0.5)
        opt_state = opt_init(params)

        update = return_update(objective, opt_update, get_params)

        opt_LL_list = []
        for i in range(optimisation_steps):
            params, opt_state, LL = update(params, opt_state)
            
            opt_LL_list.append(LL)
            
            if i > optimisation_stop_length:
                if LL < opt_LL_list[i-optimisation_stop_length] + 0.5:
                    break
                
            if i % 5e2 == 0:
                print(i, LL)

        map_psnr =  psnr(sigmoid(params['x']), image_np, smax=1)
        test_opt_psnr_mat[n_im, n_opt] = map_psnr
        test_result_dict[f'im_{n_im}_opt_{n_opt}_x'] = sigmoid(params['x'])
        test_result_dict[f'im_{n_im}_opt_{n_opt}_psnr'] = map_psnr


    # optimise from true optima to find best optima of objective 

    params = {}
    params['x'] = jax.scipy.special.logit(image_np.clip(min=1e-3, max=1-1e-3))  # sample from marginal distribution of data roughly
    params['AR_p'] = jnp.zeros(1) * 0

    objective = optimisation_objective_gen(observation, best_lambd, best_std, op_mat)

    opt_init, opt_update, get_params = optimizers.adam(optimisation_stepsize) #optimizers.momentum(step_size, mass=0.5)
    opt_state = opt_init(params)

    update = return_update(objective, opt_update, get_params)

    opt_LL_list = []
    for i in range(optimisation_steps):
        params, opt_state, LL = update(params, opt_state)
        
        opt_LL_list.append(LL)
        
        if i > optimisation_stop_length:
            if LL < opt_LL_list[i-optimisation_stop_length] + 0.5:
                break
            
        if i % 5e2 == 0:
            print(i, LL)
    
    map_psnr =  psnr(sigmoid(params['x']), image_np, smax=1)

    goodstart_opt_psnr_list.append(map_psnr)

    test_result_dict[f'im_{n_im}_opt_true_x'] = sigmoid(params['x'])
    test_result_dict[f'im_{n_im}_opt_true_psnr'] = map_psnr


test_result_dict[f'sample_psnr_list'] = test_psnr_list
test_result_dict[f'sample_psnr_list_mean'] = np.mean(test_psnr_list)
test_result_dict[f'sample_psnr_list_std'] = np.std(test_psnr_list)

test_result_dict[f'sample_LL_list'] = test_LL_list
test_result_dict[f'sample_LL_list_mean'] = np.mean(test_LL_list)
test_result_dict[f'sample_LL_list_std'] = np.std(test_LL_list)

test_result_dict[f'opt_psnr_mat'] = test_opt_psnr_mat
test_result_dict[f'opt_psnr_mean'] = np.mean(test_opt_psnr_mat)
test_result_dict[f'opt_psnr_std'] = np.std(test_opt_psnr_mat)

test_result_dict[f'opt_truestart_psnr_mat'] = goodstart_opt_psnr_list
test_result_dict[f'opt_truestart_psnr_mean'] = np.mean(goodstart_opt_psnr_list)
test_result_dict[f'opt_truestart_psnr_std'] = np.std(goodstart_opt_psnr_list)


# %% save results

with open(savedir + '/test_result_dict.pickle', 'wb') as handle:
    pickle.dump(test_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
