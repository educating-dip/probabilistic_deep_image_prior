import os
import numpy as np
import yaml
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RUN_FILE = '/localdata/jleuschn/experiments/dip_bayesian_ext/kmnist_refined_tv_strength.yaml'
RUN_FILE_APPROX = '/localdata/jleuschn/experiments/dip_bayesian_ext/runs_kmnist_approx_exact_density_block_sizes.yaml'
DIR_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext'

def get_approx_log_prob(run_path, num_samples, ddof=0):
    log_probs = []
    for i in range(num_samples):
        predictive_image_log_prob_dict = torch.load(os.path.join(run_path, 'predictive_image_log_prob_{}.pt'.format(i)), map_location='cpu')
        assert len(predictive_image_log_prob_dict['block_log_probs']) == 1  # one single block
        num_pixels = np.prod(predictive_image_log_prob_dict['block_mask_inds'][0].shape)
        log_probs.append(predictive_image_log_prob_dict['approx_log_prob'].item() / num_pixels)
    return {
        'mean': np.mean(log_probs),
        'std_error': np.std(log_probs, ddof=ddof) / np.sqrt(num_samples)}

def get_exact_log_probs(run_path, num_samples, ddof=0):
    log_probs_mll = []
    log_probs_tv_map = []
    for i in range(num_samples):
        test_log_lik_dict = np.load(os.path.join(run_path, 'test_log_lik_info_{}.npz'.format(i)), allow_pickle=True)['test_log_lik'].item()
        log_probs_mll.append(test_log_lik_dict['test_loglik_MLL'])
        log_probs_tv_map.append(test_log_lik_dict['test_loglik_type-II-MAP'])
    return {'mll': {
                'mean': np.mean(log_probs_mll),
                'std_error': np.std(log_probs_mll, ddof=ddof) / np.sqrt(num_samples)},
            'tv_map': {
                'mean': np.mean(log_probs_tv_map),
                'std_error': np.std(log_probs_mll, ddof=ddof) / np.sqrt(num_samples)}}

if __name__ == '__main__':

    num_samples = 5

    mean_log_probs_approx = {}
    mean_log_probs_exact = {}

    noise_list = [0.05, 0.1]

    highlight_idx = 3

    vec_batch_size_list = [25, 10, 5]

    with open(RUN_FILE, 'r') as f:
        runs_dict = yaml.safe_load(f)

    with open(RUN_FILE_APPROX, 'r') as f:
        runs_dict_approx = yaml.safe_load(f)

    angles = 20
    for noise in noise_list:
        mean_log_probs_approx[(angles, noise)] = {}
        mean_log_probs_exact[(angles, noise)] = {}

        for vec_batch_size in vec_batch_size_list:
            mean_log_probs_approx[(angles, noise)][vec_batch_size] = {}
            for method in ['mll', 'tv_map']:
                run_path = runs_dict_approx[angles][noise][vec_batch_size]['no_predcp' if method == 'mll' else 'predcp']
                if '/multirun/' in run_path:
                    run_path = os.path.join(DIR_PATH, 'multirun', run_path.split('/multirun/')[-1], '0')  # translate to local path, choose first sub-run
                else:
                    run_path = os.path.join(DIR_PATH, 'outputs', run_path.split('/outputs/')[-1])  # translate to local path
                cfg = OmegaConf.load(os.path.join(run_path, '.hydra', 'config.yaml'))
                job_name = OmegaConf.load(os.path.join(run_path, '.hydra', 'hydra.yaml')).hydra.job.name
                if job_name == 'exact_density_for_bayes_dip':
                    load_path = cfg.density.compute_single_predictive_cov_block.load_path
                    cfg = OmegaConf.load(os.path.join(load_path, '.hydra', 'config.yaml'))
                    job_name = OmegaConf.load(os.path.join(load_path, '.hydra', 'hydra.yaml')).hydra.job.name
                assert cfg.beam_num_angle == angles
                assert cfg.noise_specs.stddev == noise
                assert cfg.mrglik.optim.include_predcp == (method == 'tv_map')
                assert cfg.mrglik.impl.vec_batch_size == vec_batch_size
                mean_log_probs_approx[(angles, noise)][vec_batch_size][method] = get_approx_log_prob(run_path, num_samples=num_samples)

        run_path_exact = runs_dict[angles][noise]
        if '/multirun/' in run_path_exact:
            run_path_exact = os.path.join(DIR_PATH, 'multirun', run_path_exact.split('/multirun/')[-1], '0')  # translate to local path, choose first sub-run
        else:
            run_path_exact = os.path.join(DIR_PATH, 'outputs', run_path_exact.split('/outputs/')[-1])  # translate to local path
        cfg = OmegaConf.load(os.path.join(run_path_exact, '.hydra', 'config.yaml'))
        assert cfg.beam_num_angle == angles
        assert cfg.noise_specs.stddev == noise
        mean_log_probs_exact[(angles, noise)] = get_exact_log_probs(run_path_exact, num_samples=num_samples)

        print((angles, noise))
        print('approx', mean_log_probs_approx[(angles, noise)])
        print('exact', mean_log_probs_exact[(angles, noise)])

    table = ''
    for noise in noise_list:
        table += ' & ' + '\multicolumn{{2}}{{c}}{{\\textbf{{{}\,\% noise}}}}'.format(int(noise * 100))
    table += '\\\\\n'
    for noise in noise_list:
        table += ' & MLL & TV-MAP'
    table += '\\\\\n'

    table += 'exact'
    for noise in noise_list:
        table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_probs_exact[(angles, noise)]['mll']['mean'], mean_log_probs_exact[(angles, noise)]['mll']['std_error'])
        table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_probs_exact[(angles, noise)]['tv_map']['mean'], mean_log_probs_exact[(angles, noise)]['tv_map']['std_error'])
    table += '\\\\\n'

    for vec_batch_size in vec_batch_size_list:
        table += '{} probe vectors'.format(vec_batch_size)
        for noise in noise_list:
            mean_log_probs_approx_per_method = mean_log_probs_approx[(angles, noise)][vec_batch_size]
            table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_probs_approx_per_method['mll']['mean'], mean_log_probs_approx_per_method['mll']['std_error'])
            table += ' & ' + '${:.4f} \pm {:.4f}$'.format(mean_log_probs_approx_per_method['tv_map']['mean'], mean_log_probs_approx_per_method['tv_map']['std_error'])
        table += '\\\\\n'

    print(table)
