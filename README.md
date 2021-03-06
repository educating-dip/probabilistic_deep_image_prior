## A Probabilistic Deep Image Prior for Computational Tomography ([arXiv](https://arxiv.org/pdf/2203.00479.pdf))

![walnut_1x1](https://user-images.githubusercontent.com/47638906/154972964-feb149ef-1135-4eb8-b8a6-c8292ac8d172.png)

## Walnut

As a preliminary, the walnut data needs to be placed in `src/experiments/walnuts`: Download [Walnut1.zip](https://zenodo.org/record/2686726/files/Walnut1.zip?download=1) and unzip to `src/experiments/walnuts/Walnut1`. The data is described in Der Sarkissian et al., ["A Cone-Beam X-Ray CT Data Collection Designed for Machine Learning"](https://doi.org/10.1038/s41597-019-0235-y), _Sci Data_ **6**, 215 (2019); see also [the containing Zenodo record](https://zenodo.org/record/2686726/) and [this code repository](https://github.com/cicwi/WalnutReconstructionCodes).

The sparse ray transform matrix for the fan-beam like 2D sub-geometry can be downloaded from the [here](https://zenodo.org/record/6141017/files/single_slice_ray_trafo_matrix_walnut1_orbit2_ass20_css6.mat?download=1) and needs to be placed in `src/experiments/walnuts/single_slice_ray_trafo_matrix_walnut1_orbit2_ass20_css6.mat`; alternatively, it can be created with `src/experiments/create_walnut_ray_trafo_matrix.py`.

The reconstruction process with model uncertainty is split into the following steps (commands are exemplary):

### 1.  Obtain an EDIP reconstruction
```shell
python pretrain.py data=walnut trafo=walnut_ray_trafo net=walnut_unet trn.batch_size=4
python dip.py data=walnut trafo=walnut_ray_trafo net=walnut_unet net.learned_params_path=...
```

The results from such runs can be downloaded from the [supplementing material Zenodo record](https://zenodo.org/record/6141017/) ([pretrain](https://zenodo.org/record/6141017/files/walnut_pretraining.zip?download=1), [dip](https://zenodo.org/record/6141017/files/walnut_edip.zip?download=1)) and then need to be extracted to `src/experiments/outputs/` and `src/experiments/multirun/`, respectively (the paths are preconfigured in `src/cfgs/net/walnut_unet.yaml`, options `net.learned_params_path` and `net.finetuned_params_path`); alternatively, the commands above may be used to re-run the experiments.

### 2.  Run Bayes-DIP likelihood optimization
```shell
python bayes_dip.py data=walnut net=walnut_unet trafo=walnut_ray_trafo linearize_weights=True lin_params.iterations=1500 mrglik.optim.include_predcp=True mrglik.priors.clamp_variances_min_log=-6.9 mrglik.optim.tv_samples=20 mrglik.impl.vec_batch_size=10 mrglik.optim.iterations=2000
```
This runs Bayes-DIP (TV-MAP); in order to run Bayes-DIP (MLL), specify `mrglik.optim.include_predcp=False` (instead of `...=True`).

Let the output path of this step be `$OUTPUT_BAYES_DIP`.

### 3.  Assemble covariance matrix in observation space

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

### 5.  Evaluate approximate density based on samples for any patch size
```shell
python estimate_density_from_samples.py use_double=True data=walnut trafo=walnut_ray_trafo net=walnut_unet mrglik.impl.vec_batch_size=1 density.compute_single_predictive_cov_block.block_idx=walnut_inner density.block_size_for_approx=2 density.compute_single_predictive_cov_block.load_path=$OUTPUT_BAYES_DIP density.compute_single_predictive_cov_block.load_mrglik_opt_iter=1599 mrglik.priors.clamp_variances=False density.compute_single_predictive_cov_block.cov_obs_mat_load_path=$OUTPUT_ASSEMBLE_COV_OBS_MAT density.cov_obs_mat_eps_mode=abs density.cov_obs_mat_eps=0.1 density.estimate_density_from_samples.samples_load_path_list=[$OUTPUT_SAMPLES_0,$OUTPUT_SAMPLES_1]
```

We benchmark against [DIP-MCDO](https://proceedings.mlr.press/v121/laves20a.html) and the numerical analysis is reported in the table below 
<p align="center"><img src="https://user-images.githubusercontent.com/47638906/154974779-3f181dcb-7c6d-4495-8875-8ff1a25b7e1c.PNG" width="450px"></p>

## KMNIST
The full procedure is run for a single setting for 50 test samples with:
```shell
python low_rank_obs_space_mrg_lik_opt.py use_double=True net.optim.gamma=1e-4 net.optim.iterations=41000 mrglik.optim.scaling_fct=50 noise_specs.stddev=0.05 mrglik.optim.iterations=1500 beam_num_angle=20 num_images=50
```
This runs both Bayes-DIP (MLL) and Bayes-DIP (TV-MAP). For each setting (`beam_num_angle`, `noise_specs.stddev`), suitable hyperparameters (`net.optim.gamma`, `net.optim.iterations`) should be selected, which can be found in a comment in `src/experiments/eval_dip_mnist_hyper_param_search.py`.

Again, we benchmark against two probabilistic DIP formulations [DIP-MCDO](https://proceedings.mlr.press/v121/laves20a.html) and [DIP-SGLD](https://people.cs.umass.edu/~zezhoucheng/gp-dip/) and the numerical analysis is reported in the table below 
<p align="center"><img src="https://user-images.githubusercontent.com/47638906/154973023-9c70c260-776d-4ed5-aa74-3d0349a1af79.PNG" width="500px"></p>

## Setup environment and paths
We recommend using a conda environment, which can be created with `bash -i env_info/create_bayes_dip_env.sh -n <environment_name> [--cudatoolkit <cudatoolkit_version>]`.
See `env_info/bayes_dip_env.txt` for a `conda list` dump.

On a minor note, we observed higher numerical accuracy using CUDA 10 compared to CUDA 11; we recommend trying CUDA 10 instead of CUDA 11, especially for the scalable computations employing Jacobian-vector and vector-Jacobian products involving large networks.

The `src` folder is assumed to be contained in the `PYTHONPATH` environment variable.
Data paths are relative to the current working directory, which is assumed to be `src/experiments` by default.
Output paths of previous experiments are usually specified as absolute paths when passed as an option to a subsequent experiment.


## Citation

If you find this code useful, please consider citing our paper:

> Javier Antor??n, Riccardo Barbano, Johannes Leuschner, Jos?? Miguel Hern??ndez-Lobato &  Bangti Jin. (2022). A Probabilistic Deep Image Prior for Computational Tomography.

```bibtex
@misc{antoran2022bayesdip,
    title={A Probabilistic Deep Image Prior for Computational Tomography},
    author={Javier Antor??n and Riccardo Barbano and Johannes Leuschner and Jos?? Miguel Hern??ndez-Lobato and Bangti Jin},
    year={2022},
    eprint={2203.00479},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
}
``` 
