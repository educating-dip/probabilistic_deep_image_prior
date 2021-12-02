import torch
from collections.abc import Iterable

def compute_jacobian_single_batch(
    input,
    model,
    out_dim,
    return_on_cpu=False,
    ):

    jac = []
    model.eval()
    f = model(input)[0].view(-1)
    for o in range(out_dim):
        f_o = f[o]
        model.zero_grad()
        f_o.backward(retain_graph=True)
        jacs_o = agregate_flatten_weight_grad(model).detach()
        jac.append(jacs_o)
    return (torch.stack(jac,
            dim=0) if not return_on_cpu else torch.stack(jac,
            dim=0).cpu())

def agregate_flatten_weight_grad(model, include_block=['down', 'up']):

    grads_o = []
    for sect_name in include_block:
        group_blocks = getattr(model, sect_name)
        if isinstance(group_blocks, Iterable):
            for (k, block) in enumerate(group_blocks):
                for layer in block.conv:
                    if isinstance(layer, torch.nn.Conv2d):
                        grads_o.append(layer.weight.grad.flatten())
    return torch.cat(grads_o)
    