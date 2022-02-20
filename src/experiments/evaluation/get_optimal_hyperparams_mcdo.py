import sys
sys.path.append('../')

import os
import numpy as np 
from omegaconf import OmegaConf


def load_data_from_path(path):
    return np.load(os.path.join(path, 'overall_metrics9.npz'), allow_pickle=True)['avg_image_metric']

DIRPATH='src/experiments/evaluation/'  # TODO insert absolute path if needed
runs = OmegaConf.load(os.path.join(DIRPATH, 'grid_search_runs_mcdo.yaml'))
# gathering data
for num_angles in [5, 10, 20, 30]:
    for stddev in [0.05, 0.1]:
        optimal_p = {}
        for p in [0.05, 0.1, 0.2]:
            path = runs[num_angles][stddev][p]
            psnr = load_data_from_path(path)
            optimal_p[p] = psnr
        print('({}, {}): ({})'.format(num_angles, stddev, max(optimal_p, key=optimal_p.get)))

