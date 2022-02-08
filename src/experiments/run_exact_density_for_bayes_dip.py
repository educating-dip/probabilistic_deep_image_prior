import os
import yaml

RUN_FILE = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx.yaml'
DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext'

with open(RUN_FILE, 'r') as f:
    runs_dict = yaml.safe_load(f)

num_images = 5

for angles in [20]:
    for noise in [0.05, 0.1]:
        for vec_batch_size in [5, 10, 25]:
            for method in ['predcp', 'no_predcp']:
                run_path = runs_dict[angles][noise][vec_batch_size][method]
                run_path = os.path.join(DIR_PATH, 'outputs', run_path.split('/outputs/')[-1])  # translate to local path
                print(f'python src/experiments/exact_density_for_bayes_dip.py --multirun use_double=True density.compute_single_predictive_cov_block.load_path={run_path} mrglik.priors.clamp_variances=False beam_num_angle={angles} noise_specs.stddev={noise} density.block_size_for_approx=28,14,7,4,2,1 num_images={num_images}')
