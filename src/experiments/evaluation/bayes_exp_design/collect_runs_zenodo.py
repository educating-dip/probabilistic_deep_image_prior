import os
from re import L
import yaml
from omegaconf import OmegaConf

OUT_PATH = '/localdata/jleuschn/experiments/dip_bayesian_ext/zenodo'

RUN_FILE = '/home/jleuschn/dip_bayesian_ext/src/experiments/evaluation/bayes_exp_design/rectangles_runs.yaml'

with open(RUN_FILE, 'r') as f:
    runs = yaml.safe_load(f)

runs_baseline = {}
runs_random = {}
runs_priors = {}

for noise in [0.05, 0.1]:
    for prior in ['mll_prior', 'mll_prior_refinement', 'g_prior', 'g_prior_refinement', 'linear_model_isotropic', 'linear_model_gp', 'baseline', 'random']:
        if prior == 'baseline':
            for recon_method in ['dip', 'tv']:
                if recon_method in runs[noise][prior]:
                    runs_baseline.setdefault(noise, {})
                    runs_baseline[noise][recon_method] = runs[noise][prior][recon_method]
            # print(list(runs[noise][prior].keys()))
        elif prior == 'random':
            for recon_method in ['dip', 'tv']:
                if recon_method in runs[noise][prior]['random']:
                    runs_random.setdefault(noise, {})
                    runs_random[noise][recon_method] = runs[noise][prior]['random'][recon_method]
            # print(list(runs[noise][prior]['random'].keys()))
        else:
            for crit in ['EIG', 'diagonal_EIG', 'var']:
                if crit in runs[noise][prior]:
                    for recon_method in ['dip', 'tv']:
                        if recon_method in runs[noise][prior][crit]:
                            runs_priors.setdefault(noise, {})
                            runs_priors[noise].setdefault(prior, {})
                            runs_priors[noise][prior].setdefault(crit, {})
                            runs_priors[noise][prior][crit][recon_method] = runs[noise][prior][crit][recon_method]
                    # print(list(runs[noise][prior][crit]))

def translate_path(p):
    translations = {
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/javi_results/dip_mll_designs/outputs/': '/localdata/jleuschn/experiments/dip_bayesian_ext/dip_mll_designs/outputs/',
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/javi_results/g_prior_designs/outputs/': '/localdata/jleuschn/experiments/dip_bayesian_ext/g_prior_designs/outputs/',
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/scripts_bayesian_linear_model/outputs/': '/localdata/jleuschn/experiments/dip_bayesian_ext/hpc_outputs_linear/',
    }
    translations_hpc = {
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/javi_results/dip_mll_designs/outputs/': '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/experiments_results_files/bayesian_experiemental_deasign_linear_model_results/dip_mll_designs/outputs/',
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/javi_results/g_prior_designs/outputs/': '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/experiments_results_files/bayesian_experiemental_deasign_linear_model_results/g_prior_designs/outputs/',
            '/localdata/jleuschn/experiments/dip_bayesian_ext/linear_designs/': '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/experiments_results_files/bayesian_experiemental_deasign_linear_model_results/linear_designs/',
            '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/scripts_bayesian_linear_model/outputs/': '/home/rb876/rds/rds-t2-cs133-hh9aMiOkJqI/dip/experiments_results_files/bayesian_experimental_design_linear_model/outputs/',
    }

    if ':/' in p:
        server, path = p.split(':', maxsplit=1)
    else:
        path = p
    path = path.rstrip('/')
    for t_orig, t_new in translations.items():
        if path.startswith(t_orig):
            path = path.replace(t_orig, t_new, 1)
    return path

noise_dict = {0.05: 'noise05', 0.1: 'noise10'}
crit_dict = {'var': 'ESE', 'EIG': 'EIG', 'diagonal_EIG': 'diagonal_EIG'}

for noise in [0.05, 0.1]:
    path_baseline_0 = runs_baseline[noise][list(runs_baseline[noise].keys())[0]][0]
    path_baseline_0 = translate_path(path_baseline_0)
    cfg_baseline_0 = OmegaConf.load(os.path.join(path_baseline_0, '.hydra', 'config.yaml'))
    path_init_mll = cfg_baseline_0.density.compute_single_predictive_cov_block.load_path
    base_path_init_mll = os.path.join(OUT_PATH, f'initial_mll_{noise_dict[noise]}')
    path_init_mll = translate_path(path_init_mll)
    os.makedirs(base_path_init_mll, exist_ok=True)
    os.symlink(path_init_mll, os.path.join(base_path_init_mll, os.path.basename(path_init_mll)))

    for recon_method in runs_baseline[noise].keys():
        path_list_baseline = runs_baseline[noise][recon_method]
        base_path_baseline = os.path.join(OUT_PATH, f'equidistant_baseline_{noise_dict[noise]}', recon_method)
        os.makedirs(base_path_baseline, exist_ok=True)
        for path in path_list_baseline:
            path = translate_path(path)
            dest_path = os.path.join(base_path_baseline, os.path.basename(path))
            try:
                os.symlink(path, dest_path)
            except FileExistsError:
                print(dest_path, 'already exists')
            cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
            assert cfg.bed.use_best_inds_from_path is None

    for recon_method in runs_random[noise].keys():
        base_path_random_angle_selection = os.path.join(OUT_PATH, f'random_baseline_{noise_dict[noise]}', 'angle_selection')
        path_list_random = runs_random[noise][recon_method]
        base_path_random = os.path.join(OUT_PATH, f'random_baseline_{noise_dict[noise]}', recon_method)
        os.makedirs(base_path_random, exist_ok=True)
        for path in path_list_random:
            path = translate_path(path)
            dest_path = os.path.join(base_path_random, os.path.basename(path))
            try:
                os.symlink(path, dest_path)
            except FileExistsError:
                print(dest_path, 'already exists')
            cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
            if cfg.bed.use_best_inds_from_path is not None:
                os.makedirs(base_path_random_angle_selection, exist_ok=True)
                path_angle_inds = translate_path(cfg.bed.use_best_inds_from_path)
                dest_path_angle_inds = os.path.join(base_path_random_angle_selection, os.path.basename(path_angle_inds))
                try:
                    os.symlink(path_angle_inds, dest_path_angle_inds)
                except FileExistsError:
                    print(dest_path_angle_inds, 'already exists')



    for prior in ['mll_prior', 'mll_prior_refinement', 'g_prior', 'g_prior_refinement', 'linear_model_isotropic', 'linear_model_gp']:
        for crit in runs_priors[noise][prior].keys():
            base_path_prior_angle_selection = os.path.join(OUT_PATH, f'{prior}_{crit_dict[crit]}_{noise_dict[noise]}', 'angle_selection')
            for recon_method in runs_priors[noise][prior][crit].keys():
                path_list_prior = runs_priors[noise][prior][crit][recon_method]
                base_path_prior = os.path.join(OUT_PATH, f'{prior}_{crit_dict[crit]}_{noise_dict[noise]}', recon_method)
                os.makedirs(base_path_prior, exist_ok=True)
                for path in path_list_prior:
                    path = translate_path(path)
                    dest_path = os.path.join(base_path_prior, os.path.basename(path))
                    try:
                        os.symlink(path, dest_path)
                    except FileExistsError:
                        print(dest_path, 'already exists')
                    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))
                    if cfg.bed.use_best_inds_from_path is not None:
                        os.makedirs(base_path_prior_angle_selection, exist_ok=True)
                        path_angle_inds = translate_path(cfg.bed.use_best_inds_from_path)
                        dest_path_angle_inds = os.path.join(base_path_prior_angle_selection, os.path.basename(path_angle_inds))
                        try:
                            os.symlink(path_angle_inds, dest_path_angle_inds)
                        except FileExistsError:
                            print(dest_path_angle_inds, 'already exists')
                        cfg_angle_inds = OmegaConf.load(os.path.join(path_angle_inds, '.hydra', 'config.yaml'))
                        assert cfg_angle_inds.bed.get('use_best_inds_from_path', None) is None

            # remove paths from 'dip' (or 'tv') that are already in 'angle_selection'
            for recon_method in runs_priors[noise][prior][crit].keys():
                path_list_prior = runs_priors[noise][prior][crit][recon_method]
                base_path_prior = os.path.join(OUT_PATH, f'{prior}_{crit_dict[crit]}_{noise_dict[noise]}', recon_method)
                for path in path_list_prior:
                    path = translate_path(path)
                    if os.path.exists(os.path.join(base_path_prior_angle_selection, os.path.basename(path))):
                        os.remove(os.path.join(base_path_prior, os.path.basename(path)))
                        with open(os.path.join(base_path_prior, os.path.basename(path) + '_see_angle_selection'), mode='a'):
                            pass
