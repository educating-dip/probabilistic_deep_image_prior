import os
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

PATHS = [
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T12:15:05.450217Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T12:43:27.211569Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T13:11:52.646926Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T13:13:57.578716Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T13:49:29.095156Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T14:20:18.300215Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T14:50:12.565194Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T15:19:45.034374Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T15:48:46.444422Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T16:17:26.241923Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T16:46:05.178784Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T17:14:55.720979Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T17:43:12.975335Z',
    '/localdata/jleuschn/experiments/dip_bayesian_ext/outputs/2022-05-20T18:08:51.813402Z',
]

NUM_GP_PRIORS = 11
NUM_NORMAL_PRIORS = 3

FILENAME = 'jac_singular_values_per_layer'

plt.rcParams.update({
  "text.usetex": True,
})

fig, ax = plt.subplots()

gp_sort_index = [10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
gp_titles = ['\\texttt{In}'] + ['\\texttt{Down}'] * 5 + ['\\texttt{Up}'] * 5

normal_sort_index = [0, 1, 2]
normal_titles = ['\\texttt{Skip}', '\\texttt{Skip}', '\\texttt{Out}']

singular_values_list_sorted_gp = [None] * NUM_GP_PRIORS
singular_values_list_sorted_normal = [None] * NUM_NORMAL_PRIORS

label_list_sorted_gp = [None] * NUM_GP_PRIORS
label_list_sorted_normal = [None] * NUM_NORMAL_PRIORS

for path in PATHS:
    cfg = OmegaConf.load(os.path.join(path, '.hydra', 'config.yaml'))

    kind = 'normal' if len(cfg.density.exclude_gp_priors_list) == NUM_GP_PRIORS else 'gp'
    assert len(cfg.density.exclude_gp_priors_list) == NUM_GP_PRIORS - (1 if kind == 'gp' else 0)
    assert len(cfg.density.exclude_normal_priors_list) == NUM_NORMAL_PRIORS - (1 if kind == 'normal' else 0)

    if kind == 'gp':
        idx = next(i for i in range(NUM_GP_PRIORS) if i not in cfg.density.exclude_gp_priors_list)
        sorted_idx = gp_sort_index.index(idx)
    elif kind == 'normal':
        idx = next(i for i in range(NUM_NORMAL_PRIORS) if i not in cfg.density.exclude_normal_priors_list)
        sorted_idx = normal_sort_index.index(idx)

    label = '{} {} ({})'.format('GP' if kind == 'gp' else 'Normal', idx, (gp_titles if kind == 'gp' else normal_titles)[sorted_idx])

    d = torch.load(os.path.join(path, 'low_rank_jac.pt'), map_location='cpu')
    singular_values = d['S']

    if kind == 'gp':
        singular_values_list_sorted_gp[sorted_idx] = singular_values
        label_list_sorted_gp[sorted_idx] = label
    elif kind == 'normal':
        singular_values_list_sorted_normal[sorted_idx] = singular_values
        label_list_sorted_normal[sorted_idx] = label

for singular_values, label in zip(singular_values_list_sorted_gp, label_list_sorted_gp):
    if singular_values is not None:
        plt.plot(singular_values, label=label)
for singular_values, label in zip(singular_values_list_sorted_normal, label_list_sorted_normal):
    if singular_values is not None:
        plt.plot(singular_values, label=label)

ax.legend()
fig.savefig(FILENAME + '.png', dpi=600)
fig.savefig(FILENAME + '.pdf')

ax.set_yscale('log')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.savefig(FILENAME + '_log.png', dpi=600, bbox_inches='tight')
fig.savefig(FILENAME + '_log.pdf', bbox_inches='tight')
