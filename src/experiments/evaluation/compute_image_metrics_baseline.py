import sys
sys.path.append('../')

import os
import numpy as np 
import bios
import warnings
from deep_image_prior.utils import PSNR, SSIM
from omegaconf import OmegaConf

def load_data_from_path(path, idx):
    return np.load(os.path.join(path, 'recon_info_{}.npz'.format(idx)), allow_pickle=True)

def extract_recon_from_data(path, idx):
    # this extracts test-log lik. for test_loglik_MLL (w/o predcp), test_loglik_type-II-MAP (w predcp) & 2 det. baselines
    recon_info = load_data_from_path(path, idx)
    return recon_info['recon'], recon_info['image']

def compute_test_image_metrics(path, num_images):
    psnrs, ssims = [], []
    for idx in range(num_images):
        recon, image = extract_recon_from_data(path, idx)
        if recon.shape != image.shape:
            recon = recon.reshape(*image.shape)
        psnrs.append(PSNR(recon.squeeze(), image.squeeze()))
        ssims.append(SSIM(recon.squeeze(), image.squeeze()))
    return np.mean(psnrs), np.mean(ssims)

DIRPATH='src/experiments/evaluation/'  # TODO insert absolute path if needed
experiment_name = 'kmnist'

table_dict = {}
if experiment_name == 'kmnist':
    # gathering data
    for stddev in [0.05, 0.1]:
        for num_angles in [5, 10, 20, 30]:
            for run_name in ['kmnist_mcdo_baseline.yaml','kmnist_sgld_baseline_bw_005.yaml']:
                runs = OmegaConf.load(os.path.join(DIRPATH, run_name))
                path_to_data = runs[num_angles][stddev]
                exp_conf = OmegaConf.load(os.path.join(path_to_data, '.hydra/config.yaml'))
                data = compute_test_image_metrics(path_to_data, exp_conf.num_images)
                table_dict[(run_name, num_angles, stddev)] = data
    # constructing table
    rows = ['DIP-MCDO', 'DIP-SGLD']
    for stddev in [0.05, 0.1]:
        print(stddev)
        table = ''
        for row, run_name in zip(rows, ['kmnist_mcdo_baseline.yaml','kmnist_sgld_baseline_bw_005.yaml']): 
            out_row = row
            for num_angles in [5, 10, 20, 30]:
                out_row += '& ${:.2f}$/${:.3f}$ '.format(table_dict[(run_name, num_angles, stddev)][0], table_dict[(run_name, num_angles, stddev)][1])
            table += out_row + '\\\\ \n'
        print(table)
else: 
    raise NotImplementedError