# %%
import os
import sys
from absl.flags import Error
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

import numpyro 
# numpyro.set_host_device_count(5)

# module load cuda/11.4
# module load gcc/8

# %% Problem setup


img_side = 28
num_angles = 5 #10  #5
noise_std_prop = 0.1 #0.05 # 0.1

op_mat = gen_op_mat(img_side, num_angles)


kmnist_path = 'kmnist'  # TODO insert absolute path if needed

train_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=True))
test_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=False))


# %% Settings

warmup = 10000
n_samples = 10000
thinning = 2
num_chains = 5

N_images_val = 10
N_images_test = 50

# prior_std_list = [1e-2, 0.05, 1e-1, 0.5, 1, 5, 10] # switch to 1
# TV_lambda_list = [10, 50, 100, 500, 1000, 5000, 10000] # switch to 1000
# bandwidth_list = [1e-2, 0.05, 1e-1, 0.5, 1]

N_optimisation_restarts = 5
optimisation_steps = int(1e5)
optimisation_stepsize = 1e-2
reduced_optimisation_stepsize = 1e-3
optimisation_stop_length = 1000



# %% logging structures

model_mode = 'Gaussian' #'TV' #

savedir = f'../save/{model_mode}_HMC_5_01'  #10_005/'

os.makedirs(savedir, exist_ok=True)

hyperparam_search_result_dict = {}
bw_search_result_dict = {}
test_result_dict = {}


# %% Choose method to run


if model_mode == 'Gaussian':
    sampling_model = direct_model_normal_centering_prior
    optimisation_objective_gen = gen_direct_normal_centering_MLE_objective
    # parameter_vector = prior_std_list

elif model_mode == 'TV':
    sampling_model = direct_model_tv_prior
    optimisation_objective_gen = gen_direct_TV_MLE_objective
    # parameter_vector = TV_lambda_list

else:
    raise Error('Unknown mode')

# %% Find hyperparameters

# if parameter_vector is not None:

#     hyperparam_search_result_dict['parameter_vector'] = parameter_vector

#     psnr_mat = np.zeros((N_images_val, len(parameter_vector))) * np.nan

#     for n_im in range(N_images_val):
#         (example_image, _) = next(train_dset)
        
#         image_np = example_image.numpy().flatten()
#         observation = op_mat @ image_np
#         observation += np.random.randn(*observation.shape) * noise_std_prop * jnp.abs(observation).mean()
        
#         for n_param, param in enumerate(parameter_vector):
#             model = partial(sampling_model, observation, op_mat, param, noise_std_prop)

#             samples = draw_samples(model, warmup, n_samples, thinning, num_chains)
#             reconstruction_mean = samples['x'].mean(axis=0)
#             xmean_psnr = psnr(reconstruction_mean, image_np, smax=1)
#             psnr_mat[n_im, n_param] = xmean_psnr

#             hyperparam_search_result_dict[f'im_{n_im}_param_{n_param}_x'] = image_np
#             hyperparam_search_result_dict[f'im_{n_im}_param_{n_param}_y'] = observation
#             hyperparam_search_result_dict[f'im_{n_im}_param_{n_param}_samples'] = samples
#             hyperparam_search_result_dict[f'im_{n_im}_param_{n_param}_xmean_psnr'] = xmean_psnr
            
#     average_psnr = psnr_mat.mean(axis=0)
#     best_idx = np.argmax(np.round(average_psnr, decimals=2))
#     best_param = parameter_vector[best_idx]

#     hyperparam_search_result_dict['psnr_mat'] = psnr_mat
#     hyperparam_search_result_dict['best_idx'] = best_idx
#     hyperparam_search_result_dict['best_param'] = best_param


# with open(savedir + '/hyperparam_search_result_dict.pickle', 'wb') as handle:
#     pickle.dump(hyperparam_search_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% Find bandwidth
from sklearn.neighbors import KernelDensity



# LL_mat = np.zeros((N_images_val, len(bandwidth_list))) * np.nan

# bw_search_result_dict['parameter_vector'] = bandwidth_list

# for n_im in range(N_images_val):
#     (example_image, _) = next(train_dset)
        
#     image_np = example_image.numpy().flatten()
#     observation = op_mat @ image_np
#     observation += np.random.randn(*observation.shape) * noise_std_prop *  jnp.abs(observation).mean()
    
#     model = partial(sampling_model, observation, op_mat, best_param, noise_std_prop)
#     samples = draw_samples(model, warmup, n_samples, thinning, num_chains)
    
#     for bw_idx, bw in enumerate(bandwidth_list):
#         kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(samples['x']) 
#         LL = kde.score_samples(image_np[None,:]) / len(image_np)
#         LL_mat[n_im, bw_idx] = LL

#         bw_search_result_dict[f'im_{n_im}_param_{bw_idx}_x'] = image_np
#         bw_search_result_dict[f'im_{n_im}_param_{bw_idx}_y'] = observation
#         bw_search_result_dict[f'im_{n_im}_param_{bw_idx}_samples'] = samples
#         bw_search_result_dict[f'im_{n_im}_param_{bw_idx}_LL'] = LL

# mean_LL = LL_mat.mean(axis=0)
# best_idx = np.argmax(mean_LL)
# best_bw = bandwidth_list[best_idx]

# bw_search_result_dict['LL_mat'] = LL_mat
# bw_search_result_dict['best_idx'] = best_idx
# bw_search_result_dict['best_param'] = best_bw
        

# with open(savedir + '/bw_search_result_dict.pickle', 'wb') as handle:
#     pickle.dump(bw_search_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



# %% Draw test samples


best_bw = 0.1
best_param = 5#000 # note that for 10 angle 5% noise, best TV is 500. For 5 angles 10% noise it is 5000

test_psnr_list = []
test_LL_list = []

test_opt_psnr_mat = np.zeros((N_images_test, N_optimisation_restarts)) * np.nan

goodstart_opt_psnr_list = []

for n_im in range(N_images_test):

    # draw image
    (example_image, _) = next(test_dset)
    image_np = example_image.numpy().flatten()
    observation = op_mat @ image_np
    observation += np.random.randn(*observation.shape) * noise_std_prop * jnp.abs(observation).mean()
    
    # draw samples
    model = partial(sampling_model, observation, op_mat, best_param, noise_std_prop)
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

        objective = optimisation_objective_gen(observation, best_param, op_mat, noise_std_prop)

        opt_init, opt_update, get_params = optimizers.adam(optimisation_stepsize) #optimizers.momentum(step_size, mass=0.5)
        opt_state = opt_init(params)

        update = return_update(objective, opt_update, get_params)

        opt_LL_list = []
        LR_reduced = False
        for i in range(optimisation_steps):
            params, opt_state, LL = update(params, opt_state)
            
            opt_LL_list.append(LL)
            
            if i > optimisation_stop_length:
                if LL < opt_LL_list[i-optimisation_stop_length] + 0.5 and not LR_reduced:
                    print('reached plateau, reducing LR')
                    LR_reduced = True
                    _, opt_update, get_params = optimizers.adam(reduced_optimisation_stepsize)
                elif LL < opt_LL_list[i-optimisation_stop_length] + 0.05:
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

    objective = optimisation_objective_gen(observation, best_param, op_mat, noise_std_prop)

    opt_init, opt_update, get_params = optimizers.adam(optimisation_stepsize) #optimizers.momentum(step_size, mass=0.5)
    opt_state = opt_init(params)

    update = return_update(objective, opt_update, get_params)

    opt_LL_list = []
    LR_reduced = False
    for i in range(optimisation_steps):
        params, opt_state, LL = update(params, opt_state)
        
        opt_LL_list.append(LL)
        
        if i > optimisation_stop_length:
            if LL < opt_LL_list[i-optimisation_stop_length] + 0.5 and not LR_reduced:
                print('reached plateau, reducing LR')
                LR_reduced = True
                _, opt_update, get_params = optimizers.adam(reduced_optimisation_stepsize)
            elif LL < opt_LL_list[i-optimisation_stop_length] + 0.05:
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
