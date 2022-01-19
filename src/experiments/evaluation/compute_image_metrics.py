import sys
sys.path.append('../')

import os
import numpy as np 
import bios
import warnings
from deep_image_prior.utils import PSNR, SSIM
from omegaconf import OmegaConf

def display_experimental_run(runs, experiment_name):
    print('{} run/s found:'.format(len(runs[experiment_name])))
    for run in runs[experiment_name]:
        print('path: {}, num_angles: {}, stddev: {}'.format(run['path'], run['num_angles'], run['stddev']))

def load_data_from_path(path, idx):
    return np.load(os.path.join(path, 'recon_info_{}.npz'.format(idx)), allow_pickle=True), \
        np.load(os.path.join(path, 'test_log_lik_info_{}.npz'.format(idx)), allow_pickle=True)

def extract_recon_from_data(path, idx):
    # this extracts test-log lik. for test_loglik_MLL (w/o predcp), test_loglik_type-II-MAP (w predcp) & 2 det. baselines
    recon_info, _  = load_data_from_path(path, idx)
    return recon_info['recon'], recon_info['image']

def compute_test_image_metrics(path, num_images):
    psnrs, ssims = [], []
    for idx in range(num_images):
        recon, image = extract_recon_from_data(path, idx)
        psnrs.append(PSNR(recon.squeeze(), image.squeeze()))
        ssims.append(SSIM(recon.squeeze(), image.squeeze()))
    return np.mean(psnrs), np.mean(ssims)

DIRPATH='/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/dip_bayesian_ext/src/experiments/evaluation/'
experiment_name = 'kmnist'
runs = OmegaConf.load(os.path.join(DIRPATH, 'runs.yaml'))

table_dict = {}
if experiment_name == 'kmnist':
    # gathering data
    for stddev in [0.05, 0.1]:
        for num_angles in [5, 10, 20, 30]:
                path_to_data = runs.kmnist[num_angles][stddev][0]['path'] # selecting first run in yaml file [0]
                exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
                data = compute_test_image_metrics(path_to_data, exp_conf.num_images)
                table_dict[(num_angles, stddev)] = data
    # constructing table
    for stddev in [0.05, 0.1]:
        print(stddev)
        table = ''
        for num_angles in [5, 10, 20, 30]:
            table += '& ${:.2f}$/'.format(table_dict[(num_angles, stddev)][0])
            table += '${:.3f}$'.format(table_dict[(num_angles, stddev)][1])
        print(table)
else: 
    raise NotImplementedError


