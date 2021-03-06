import os
import numpy as np 
import bios
import warnings 
from omegaconf import OmegaConf


def load_data_from_path(path, idx):
    return np.load(os.path.join(path, 'recon_info_{}.npz'.format(idx)), allow_pickle=True)['test_log_likelihood']

def compute_mean_and_standard_error(data):
    return (np.mean(data), np.std(data) / np.sqrt(len(data)))

def compute_test_log_liks(path, num_images):
    list_test_log_liks = []
    for idx in range(num_images):
        test_log_lik = load_data_from_path(path, idx)
        list_test_log_liks.append(test_log_lik)
    return compute_mean_and_standard_error(list_test_log_liks)

DIRPATH='src/experiments/evaluation/'  # TODO insert absolute path if needed
experiment_name = 'kmnist'
rows = ['DIP-MCDO', 'DIP-SGLD']

table_dict = {}
if experiment_name == 'kmnist':
    # gathering data
    for stddev in [0.05, 0.1]:
        for num_angles in [5, 10, 20, 30]:
            for run_name in ['kmnist_mcdo_baseline_bw_005.yaml', 'kmnist_sgld_baseline_bw_005.yaml']:
                runs = OmegaConf.load(os.path.join(DIRPATH, run_name))
                path_to_data = runs[num_angles][stddev]
                exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
                data = compute_test_log_liks(path_to_data, exp_conf.num_images)
                table_dict[(run_name, num_angles, stddev)] = data
    # constructing table
    for stddev in [0.05, 0.1]:
        print(stddev)
        table = ''
        for row, run_name in zip(rows, ['kmnist_mcdo_baseline_bw_005.yaml', 'kmnist_sgld_baseline_bw_005.yaml']):
            out_row = row
            for num_angles in [5, 10, 20, 30]:
                out_row += '& ${:.3f} \\pm {:.3f}$'.format(*table_dict[(run_name, num_angles, stddev)])
            table += out_row + '\\\\ \n'
        print(table)
else: 
    raise NotImplementedError
