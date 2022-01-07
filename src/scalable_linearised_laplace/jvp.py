import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from linearized_laplace import compute_jacobian_single_batch, agregate_flatten_weight_grad
from linearized_weights.linearized_weights import finite_diff_JvP
from scalable_linearised_laplace.batch_ensemble import Conv2dBatchEnsemble
from scalable_linearised_laplace.fwAD import get_fwAD_model
from scalable_linearised_laplace.conv2d_fwAD import Conv2dFwAD

class FwAD_JvP_PreserveAndRevertWeightsToParameters(object):
    def __init__(self, modules):
        self.modules = modules
    def __enter__(self):
        preserve_all_weights_block_standard_parameters(self.modules)
    def __exit__(self, type, value, traceback):
        revert_all_weights_block_to_standard_parameters(self.modules)

# note: after calling fwAD_JvP or fwAD_JvP_batch_ensemble, the weights are no
# longer stored as nn.Parameters of the respective modules; to keep the original
# parameters, wrap the code in above context manager
# (``with FwAD_JvP_PreserveAndRevertWeightsToParameters(modules): ...``), or
# manually call ``preserve_all_weights_block_standard_parameters(modules)``
# before and ``revert_all_weights_block_to_standard_parameters(modules)`` after

def fwAD_JvP(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 1
    assert len(x.shape) == 4

    model.eval()

    with torch.no_grad(), fwAD.dual_level():
        set_all_weights_block_tangents(modules, vec)
        out = model(x)[0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

# note: after calling fwAD_JvP_batch_ensemble the weights nn.Parameters of the module are
# renamed and a dual tensor attribute is stored instead under the original name;
# to obtain the original parameters, call
# ``preserve_all_weights_block_standard_parameters(modules)`` before and
# ``revert_all_weights_block_to_standard_parameters(modules)`` after
def fwAD_JvP_batch_ensemble(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()

    with torch.no_grad(), fwAD.dual_level():
        set_all_weights_block_tangents_batch_ensemble(modules, vec)
        out = model(x)[0]
        JvP = fwAD.unpack_dual(out).tangent

    return JvP

def set_all_weights_block_tangents(modules, tangents):  # TODO set weight tensors from backward model (in order to not copy)?
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dFwAD)
        n_weights = layer.weight.numel()
        weight = layer.weight
        del layer.weight
        layer.weight = fwAD.make_dual(weight, tangents[n_weights_all:n_weights_all+n_weights].view_as(weight))
        n_weights_all += n_weights

def set_all_weights_block_tangents_batch_ensemble(modules, tangents):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        weight = layer.weight
        del layer.weight
        layer.weight = fwAD.make_dual(weight, tangents[:, n_weights_all:n_weights_all+n_weights].view_as(weight))
        n_weights_all += n_weights

def preserve_all_weights_block_standard_parameters(modules):
    for layer in modules:
        assert isinstance(layer, (Conv2dFwAD, Conv2dBatchEnsemble))
        assert isinstance(layer.weight, torch.nn.Parameter)
        layer._weight_primal = layer.weight

def revert_all_weights_block_to_standard_parameters(modules):
    for layer in modules:
        assert isinstance(layer, (Conv2dFwAD, Conv2dBatchEnsemble))
        del layer.weight
        layer.weight = layer._weight_primal
        del layer._weight_primal
        # remove and re-add bias parameter to obtain original parameter order
        bias = layer.bias
        del layer.bias
        layer.bias = bias

def finite_diff_JvP_batch_ensemble(x, model, vec, modules, eps=None):

    assert len(vec.shape) == 2
    assert len(x.shape) in (4, 5)

    if len(x.shape) == 4:
        x = torch.broadcast_to(x, (vec.shape[0],) + x.shape)  # insert instance dim

    model.eval()
    with torch.no_grad():
        map_weights = get_weight_block_vec_batch_ensemble(modules)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max) / (2 * v_max)

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights_block_batch_ensemble(modules, w_plus)
        f_plus = model(x)[0]

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights_block_batch_ensemble(modules, w_minus)
        f_minus = model(x)[0]

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights_block_batch_ensemble(modules, map_weights)
        return JvP

def set_all_weights_block_batch_ensemble(modules, weights):
    n_weights_all = 0
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        n_weights = np.prod(layer.weight.shape[1:])  # dim 0 is instance dim
        layer.weight.copy_(weights[:, n_weights_all:n_weights_all+n_weights].view_as(layer.weight))
        n_weights_all += n_weights

def get_weight_block_vec_batch_ensemble(modules):
    ws = []
    for layer in modules:
        assert isinstance(layer, Conv2dBatchEnsemble)
        ws.append(layer.weight.view(layer.num_instances, -1))
    return torch.cat(ws, dim=1)
