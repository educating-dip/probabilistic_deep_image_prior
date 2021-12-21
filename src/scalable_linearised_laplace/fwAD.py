from copy import copy, deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from scalable_linearised_laplace.batch_ensemble import Conv2dBatchEnsemble
from linearized_weights.conv2d_fwAD import Conv2dFwAD, conv2d_fwAD
from linearized_weights.upsample_fwAD import UpsampleFwAD
from linearized_weights.group_norm_fwAD import GroupNormFwAD

def construct_conv2d_fwAD(module):
    assert isinstance(module, nn.Conv2d)
    new_module = Conv2dFwAD(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=module.weight.device,
            dtype=module.weight.dtype)
    new_module.weight.data.copy_(module.weight.data)
    if module.bias is not None:
        new_module.bias.data.copy_(module.bias.data)
    return new_module

def construct_upsample_fwAD(module):
    assert isinstance(module, nn.Upsample)
    new_module = UpsampleFwAD(
            size=module.size,
            scale_factor=module.scale_factor,
            mode=module.mode,
            align_corners=module.align_corners)
    return new_module

def construct_group_norm_fwAD(module):
    assert isinstance(module, nn.GroupNorm)
    new_module = GroupNormFwAD(
            num_groups=module.num_groups,
            num_channels=module.num_channels,
            eps=module.eps,
            affine=module.affine,
            device=module.weight.device if module.affine else None,
            dtype=module.weight.dtype if module.affine else None)
    return new_module

def construct_conv2d_batch_ensemble_fwAD(module):
    assert isinstance(module, Conv2dBatchEnsemble)
    new_module = deepcopy(module)
    new_module.conv2d_fun = conv2d_fwAD
    return new_module

FWAD_REPLACE = (
    (nn.Conv2d, construct_conv2d_fwAD),
    (nn.Upsample, construct_upsample_fwAD),
    (nn.GroupNorm, construct_group_norm_fwAD),
    (Conv2dBatchEnsemble, construct_conv2d_batch_ensemble_fwAD),
)

# model is considered a container that can handle multi-instance input and
# output, only its submodules are replaced
def get_fwAD_model(model, return_module_mapping=False, use_copy='share_parameters'):

    assert use_copy in ['share_parameters', 'deep', True, False, None]

    shared_parameters_from_model = model if use_copy == 'share_parameters' else None

    if use_copy:
        fwAD_model = deepcopy(model)

    if return_module_mapping:

        if use_copy:
            module_copies_to_orig_mapping = {copy: old for copy, old in zip(fwAD_model.modules(), model.modules())}

        module_mapping = {}
        replace_with_fwAD_layers(fwAD_model, out_module_mapping=module_mapping,
        shared_parameters_from_model=shared_parameters_from_model)

        if use_copy:
            # translate module_copies_mapping ("copy -> new") to original modules "old -> new"
            module_mapping = {module_copies_to_orig_mapping[copy]: new for copy, new in module_mapping.items()}

        return fwAD_model, module_mapping

    else:
        replace_with_fwAD_layers(fwAD_model)
        return fwAD_model

# inspired by https://discuss.pytorch.org/t/how-to-modify-a-pretrained-model/60509/13
def replace_with_fwAD_layers(
        model,
        add_replace=(), remove_replace=(), out_module_mapping=None,
        shared_parameters_from_model=None):

    replace = tuple(t for t in FWAD_REPLACE + tuple(add_replace) if t not in remove_replace)

    for n, module in model.named_children():
        new_constructor = None
        for old, new in replace:
            if isinstance(module, old):
                new_constructor = new
        if new_constructor is not None:
            new_module = new_constructor(module)
            setattr(model, n, new_module)
            if out_module_mapping is not None:
                out_module_mapping[module] = new_module
            if shared_parameters_from_model is not None:
                for param_name, param in module.named_parameters(recurse=False):
                    delattr(new_module, param_name)
                    shared_param = getattr(getattr(shared_parameters_from_model, n), param_name)
                    setattr(new_module, param_name, shared_param)

        # replace potential children recursively
        replace_with_fwAD_layers(
                module,
                add_replace=add_replace, remove_replace=remove_replace,
                out_module_mapping=out_module_mapping,
                shared_parameters_from_model=getattr(shared_parameters_from_model, n) if shared_parameters_from_model is not None else None)
