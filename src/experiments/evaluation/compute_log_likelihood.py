import os
import numpy as np 
from omegaconf import OmegaConf

def display_experimental_run(runs, experiment_name):
    print('{} run/s found:'.format(len(runs[experiment_name])))
    for run in runs[experiment_name]:
        print('path: {}, num_angles: {}, stddev: {}'.format(run['path'], run['num_angles'], run['stddev']))

def load_data_from_path(path, idx):
    return np.load(os.path.join(path, 'recon_info_{}.npz'.format(idx)), allow_pickle=True), \
        np.load(os.path.join(path, 'test_log_lik_info_{}.npz'.format(idx)), allow_pickle=True)

def extract_log_lik_from_data(path, idx):
    # this extracts test-log lik. for test_loglik_MLL (w/o predcp), test_loglik_type-II-MAP (w predcp) & 2 det. baselines
    _, log_lik_info = load_data_from_path(path, idx)
    return log_lik_info['test_log_lik'].item()['test_loglik_MLL'], log_lik_info['test_log_lik'].item()['test_loglik_type-II-MAP'], \
        log_lik_info['test_log_lik'].item()['noise_model'], log_lik_info['test_log_lik'].item()['noise_model_unit_variance']

def compute_mean_and_standard_error(data):
    return (np.mean(data), np.std(data) / np.sqrt(len(data)))

def compute_test_log_liks(path, num_images):
    test_log_lik = {'mll': [], 'map': [], 'det.': [], 'det.unit-var': []}
    for idx in range(num_images):
        test_log_lik_mll, test_log_lik_map, det, det_unit_var = extract_log_lik_from_data(path, idx)
        test_log_lik['mll'].append(test_log_lik_mll); test_log_lik['map'].append(test_log_lik_map); test_log_lik['det.'].append(det); test_log_lik['det.unit-var'].append(det_unit_var)
    return compute_mean_and_standard_error(test_log_lik['det.unit-var']), compute_mean_and_standard_error(test_log_lik['det.']), \
        compute_mean_and_standard_error(test_log_lik['mll']), compute_mean_and_standard_error(test_log_lik['map'])

DIRPATH='/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/dip_bayesian_ext/src/experiments/evaluation/'
experiment_name = 'kmnist'
runs = OmegaConf.load(os.path.join(DIRPATH, 'kmnist_refined_tv_strength.yaml')) # 'runs.yaml'))
rows = ['DIP ($\sigma^2_y$ = 1)', 'DIP (MLL $\sigma^2_y$)', 'Bayes DIP MLL', 'Bayes DIP TV-MAP']
table_dict = {}
if experiment_name == 'kmnist':
    # gathering data
    for stddev in [0.05, 0.1]:
        for num_angles in [5, 10, 20, 30]:
                path_to_data = runs[num_angles][stddev] # kmnist[num_angles][stddev][0]['path']
                exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
                data = compute_test_log_liks(path_to_data, exp_conf.num_images)
                table_dict[(num_angles, stddev)] = data
    # constructing table
    for stddev in [0.05, 0.1]:
        print(stddev)
        table = ''
        for i, row in enumerate(rows):
            out_row = row
            for num_angles in [5, 10, 20, 30]:
                out_row += '& ${:.3f} \\pm {:.3f}$'.format(*table_dict[(num_angles, stddev)][i])
            table += out_row + '\\\\ \n'
        print(table)
else: 
    raise NotImplementedError
