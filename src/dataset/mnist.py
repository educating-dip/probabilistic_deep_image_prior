import torch

def simulate(x, ray_trafos, cfg):

    obs = ray_trafos['ray_trafo'](x)
    relative_stddev = torch.mean(torch.abs(obs))
    observation = obs + torch.zeros(*obs.shape).normal_(0, 1) \
        * relative_stddev * cfg.stddev
    filtbackproj = ray_trafos['pseudoinverse'](observation)
    return (observation, filtbackproj, x)
