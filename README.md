# Bayes-DIP

## Setup environment and paths
We recommend using a conda environment, which can be created with `bash -i env_info/create_bayes_dip_env.sh -n <environment_name> [--cudatoolkit <cudatoolkit_version>]`.
See `env_info/bayes_dip_env.txt` for a `conda list` dump.

On a minor note, we observed higher numerical accuracy using CUDA 10 compared to CUDA 11; we recommend trying CUDA 10 instead of CUDA 11, especially for the scalable computations employing Jacobian-vector and vector-Jacobian products involving large networks.

The `src` folder is assumed to be contained in the `PYTHONPATH` environment variable.
Data paths are relative to the current working directory, which is assumed to be `src/experiments` by default.
Output paths of previous experiments are usually specified as absolute paths when passed as an option to a subsequent experiment.

## KMNIST experiments
On KMNIST, exact computations are feasible.
The full procedure is run for a single setting for 50 test samples with:
```shell
python low_rank_obs_space_mrg_lik_opt.py use_double=True net.optim.gamma=1e-4 net.optim.iterations=41000 mrglik.optim.scaling_fct=50 noise_specs.stddev=0.05 mrglik.optim.iterations=1500 beam_num_angle=20 num_images=50
```

This runs both Bayes-DIP (MLL) and Bayes-DIP (TV-MAP). For each setting (`beam_num_angle`, `noise_specs.stddev`), suitable hyperparameters (`net.optim.gamma`, `net.optim.iterations`) should be selected, which can be found in a comment in `src/experiments/eval_dip_mnist_hyper_param_search.py`.

The KMNIST data is downloaded automatically.

## Walnut experiments
On the Walnut data, a scalable approach involving approximations is needed.

As a preliminary, the walnut data needs to be placed in `src/experiments/walnuts`: Download [Walnut1.zip](https://zenodo.org/record/2686726/files/Walnut1.zip?download=1) and unzip to `src/experiments/walnuts/Walnut1`. The data is described in Der Sarkissian et al., ["A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning"](https://doi.org/10.1038/s41597-019-0235-y), _Sci Data_ **6**, 215 (2019); see also [the containing Zenodo record](https://zenodo.org/record/2686726/) and [this code repository](https://github.com/cicwi/WalnutReconstructionCodes).

The sparse ray transform matrix for the fan-beam like 2D sub-geometry is included in this repository and can be used as is (`src/experiments/walnuts/single_slice_ray_trafo_matrix_walnut1_orbit2_ass20_css6.mat`, created with `src/experiments/create_walnut_ray_trafo_matrix.py`).

The process is split into the following steps (commands are exemplary):

### 1.  Obtain an EDIP reconstruction
```shell
python pretrain.py data=walnut trafo=walnut_ray_trafo net=walnut_unet trn.batch_size=4
python dip.py data=walnut trafo=walnut_ray_trafo net=walnut_unet net.learned_params_path=...
```

The results from such runs are included in this repository under `src/experiments/` and preconfigured in `src/cfgs/net/walnut_unet.yaml` (options `net.learned_params_path` and `net.finetuned_params_path`).

### 2.  Run Bayes-DIP likelihood optimization
```shell
python bayes_dip.py data=walnut net=walnut_unet trafo=walnut_ray_trafo linearize_weights=True lin_params.iterations=1500 mrglik.optim.include_predcp=True mrglik.priors.clamp_variances_min_log=-6.9 mrglik.optim.tv_samples=20 mrglik.impl.vec_batch_size=10 mrglik.optim.iterations=2000
```

This runs Bayes-DIP (TV-MAP); in order to run Bayes-DIP (MLL), specify `mrglik.optim.include_predcp=False` (instead of `...=True`).

Let the output path of this step be `$OUTPUT_BAYES_DIP`.

### 3.  Assemble covariance matrix in observation space (`K_yy`)
```shell
python assemble_cov_obs_mat.py data=walnut trafo=walnut_ray_trafo net=walnut_unet use_double=True mrglik.impl.vec_batch_size=1 mrglik.priors.clamp_variances=False density.assemble_cov_obs_mat.load_path=$OUTPUT_BAYES_DIP density.assemble_cov_obs_mat.load_mrglik_opt_iter=1599
```

Let the output path of this step be `$OUTPUT_ASSEMBLE_COV_OBS_MAT`.

### 4.  Generate samples from the predictive posterior
```shell
python estimate_density_from_samples.py use_double=True data=walnut trafo=walnut_ray_trafo net=walnut_unet mrglik.impl.vec_batch_size=1 density.compute_single_predictive_cov_block.block_idx=walnut_inner density.block_size_for_approx=2 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP density.compute_single_predictive_cov_block.load_mrglik_opt_iter=1599 mrglik.priors.clamp_variances=False density.compute_single_predictive_cov_block.cov_obs_mat_load_path=$OUTPUT_ASSEMBLE_COV_OBS_MAT density.cov_obs_mat_eps_mode=abs density.cov_obs_mat_eps=0.1 density.num_mc_samples=8192 density.estimate_density_from_samples.seed=100
```

The sampling can be run in parallel, choosing a different seed (`density.estimate_density_from_samples.seed`) for each run. The saved samples from all runs can then be loaded in the next step.

Let the output paths of this step be `$OUTPUT_SAMPLES_0` and `$OUTPUT_SAMPLES_1`.

### 5.  Evaluate approximate density based on samples for any patch size (option `density.block_size_for_approx`)
```shell
python estimate_density_from_samples.py use_double=True data=walnut trafo=walnut_ray_trafo net=walnut_unet mrglik.impl.vec_batch_size=1 density.compute_single_predictive_cov_block.block_idx=walnut_inner density.block_size_for_approx=2 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP density.compute_single_predictive_cov_block.load_mrglik_opt_iter=1599 mrglik.priors.clamp_variances=False density.compute_single_predictive_cov_block.cov_obs_mat_load_path=$OUTPUT_ASSEMBLE_COV_OBS_MAT density.cov_obs_mat_eps_mode=abs density.cov_obs_mat_eps=0.1 density.estimate_density_from_samples.samples_load_path_list=[$OUTPUT_SAMPLES_0,$OUTPUT_SAMPLES_1]
```
