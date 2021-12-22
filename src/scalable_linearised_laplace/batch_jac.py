from .utils_batch_jac import aggregate_flatten_weight_batch_grad, clear_grads

def vec_jac_mul_batch(hooked_model, filtbackproj, v, bayesianized_model):
    
    batch_size = v.shape[0]
    assert v.shape[1] == len(filtbackproj.flatten())
    filtbackproj = filtbackproj.repeat(batch_size, 1, 1, 1) # TODO: https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html 
    hooked_model.eval()
    f = hooked_model(filtbackproj)[0].view(batch_size, -1)
    hooked_model.zero_grad()
    assert f.shape == v.shape
    f.backward(v) # (f * v).sum(dim=-1).sum(dim=0).backward()
    v_jac_mul = aggregate_flatten_weight_batch_grad(batch_size=batch_size, modules=bayesianized_model.get_all_modules_under_prior(), store_device=filtbackproj.device).detach()
    clear_grads()
    assert v_jac_mul.shape == (batch_size, bayesianized_model.num_params_under_priors)
    return v_jac_mul