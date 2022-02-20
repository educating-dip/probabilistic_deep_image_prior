# %%
import argparse

import os
import sys
N_up = 1
nb_dir = '/'.join(os.getcwd().split('/')[:-N_up] )
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
    print(nb_dir)


parser = argparse.ArgumentParser(description='Run hybrid inference in pixels space')
parser.add_argument('--N_image', type=int, required=True,
                    help='number of image to evaluate with')
parser.add_argument('--TV_lambda', type=float, default=1000.,
                    help='strength of TV prior (default: 1e3)')
parser.add_argument('--Gaussian_var', type=float, default=1.,
                    help='marginal prior variance (default: 1)')

parser.add_argument('--angles', type=int, default=5,
                    help='number of projection angles (default: 5)')
parser.add_argument('--noise_perc', type=float, default=0.1,
                    help='proportion of noise (default: 0.1)')

parser.add_argument('--results_folder', type=str, required=True,
                    help='where to store the results (default: ~/blt2/results/lenet_mnist)')

args = parser.parse_args()

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

import numpyro 
import jax
print(jax.local_device_count())
# numpyro.set_host_device_count(5)

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
num_angles = args.angles  #5
noise_std_prop = args.noise_perc # 0.1

op_mat = gen_op_mat(img_side, num_angles)

kmnist_path = 'kmnist'  # TODO insert absolute path if needed
# kmnist_path = './data/kmnist'

train_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=True))
test_dset = iter(load_KMNIST_dataset(kmnist_path, batchsize=1, train=False))


# %% Settings

warmup = 2500 # 10000
n_samples = 10000 
thinning = 2 
num_chains = 5 

# N_images_val = 10
# N_images_test = 20

# prior_std_list = [1e-2, 0.05, 1e-1, 0.5, 1, 5, 10]
# TV_lambda_list = [10, 50, 100, 500, 1000, 5000, 10000]
# bandwidth_list = [1e-2, 0.05, 1e-1, 0.5, 1]

N_optimisation_restarts = 5
optimisation_steps = int(1e5) # 1e5
optimisation_stepsize = 1e-2
reduced_optimisation_stepsize = 1e-3
optimisation_stop_length = 1000


# %% logging structures

exp_name = f'Hybrid_HMC_{num_angles}_{noise_std_prop}'
savedir = os.path.join(args.results_folder, exp_name)
os.makedirs(savedir, exist_ok=True)

hyperparam_search_result_dict = {}
bw_search_result_dict = {}
test_result_dict = {}


# %% Choose method to run

sampling_model = direct_model_predcp_tv_prior
optimisation_objective_gen = gen_direct_predcp_TV_MLE_objective

best_std = args.Gaussian_var
best_bw = 0.1
best_lambd = args.TV_lambda #500 # 5000 # note that for 10 angle 5% noise, best TV is 500. For 5 angles 10% noise it is 5000s


# %% Draw test samples

test_psnr_list = []
test_LL_list = []

test_opt_psnr_vec = np.zeros( N_optimisation_restarts) * np.nan

goodstart_opt_psnr_list = []

for n_im in range(args.N_image+1):
    (example_image, _) = next(test_dset)
# draw image


image_np = example_image.numpy().flatten()
observation = op_mat @ image_np
observation += np.random.randn(*observation.shape) * noise_std_prop * jnp.abs(observation).mean()

    # draw samples
# init_dict = {'x': sigmoid(params['x']), 'AR_p': sigmoid(params['AR_p'])}
model = partial(sampling_model, observation, op_mat, best_lambd, best_std, noise_std_prop)
samples = draw_samples(model, warmup, n_samples, thinning, num_chains, init_dict={})

reconstruction_mean = samples['x'].mean(axis=0)
xmean_psnr = psnr(reconstruction_mean, image_np, smax=1)
kde = KernelDensity(kernel='gaussian', bandwidth=best_bw).fit(samples['x']) 
LL = kde.score_samples(image_np[None,:]) / len(image_np)

test_psnr_list.append(xmean_psnr)
test_LL_list.append(LL)

print('LL', LL)

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

    objective = optimisation_objective_gen(observation, best_lambd, best_std, op_mat, noise_std_prop)

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
    print('optim map psnr', map_psnr)
    test_opt_psnr_vec[n_opt] = map_psnr
    test_result_dict[f'im_{n_im}_opt_{n_opt}_x'] = sigmoid(params['x'])
    test_result_dict[f'im_{n_im}_opt_{n_opt}_psnr'] = map_psnr


# optimise from true optima to find best optima of objective 
params = {}
params['x'] = jax.scipy.special.logit(image_np.clip(min=1e-3, max=1-1e-3))  # sample from marginal distribution of data roughly
params['AR_p'] = jnp.zeros(1) * 0

objective = optimisation_objective_gen(observation, best_lambd, best_std, op_mat, noise_std_prop)

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


############# Global metrics here ##############

# samples
test_result_dict[f'sample_psnr_list'] = test_psnr_list
test_result_dict[f'sample_psnr_list_mean'] = np.mean(test_psnr_list)
test_result_dict[f'sample_psnr_list_std'] = np.std(test_psnr_list)

test_result_dict[f'sample_LL_list'] = test_LL_list
test_result_dict[f'sample_LL_list_mean'] = np.mean(test_LL_list)
test_result_dict[f'sample_LL_list_std'] = np.std(test_LL_list)

# optimisations

test_result_dict[f'opt_psnr_vec'] = test_opt_psnr_vec
test_result_dict[f'opt_psnr_mean'] = np.mean(test_opt_psnr_vec)
test_result_dict[f'opt_psnr_std'] = np.std(test_opt_psnr_vec)

# true optimisations

test_result_dict[f'opt_truestart_psnr_mat'] = goodstart_opt_psnr_list
test_result_dict[f'opt_truestart_psnr_mean'] = np.mean(goodstart_opt_psnr_list)
test_result_dict[f'opt_truestart_psnr_std'] = np.std(goodstart_opt_psnr_list)


# %% save results

with open(savedir + f'/test_result_dict_{args.N_image}.pickle', 'wb') as handle:
    pickle.dump(test_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
