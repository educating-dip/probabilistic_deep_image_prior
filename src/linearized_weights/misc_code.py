N_epochs = 100

w_init = torch.zeros_like(w_map)

model.eval()

lin_w_fd = nn.Parameter(w_init.clone()).cuda()

lr = 1e-4

prior_prec = 1e-4 * Ntrain
wd = prior_prec / Ntrain

print('prior_prec', prior_prec, 'weight decay', wd)

optimizer = torch.optim.Adam([lin_w_fd], lr, weight_decay=0)


nll_func = nn.CrossEntropyLoss(reduction='mean')

loss_vec_fd = []


for i in range(1000):

    # get projection vector
    lin_pred = finite_diff_JvP(x, reconstructor.model, lin_w_fd.detach(), None, None).detach()
    loss = nll_func(lin_pred, y).detach() # implicit averaging across batch (squared error + TV)
    v = log_softmaxCE_grad(lin_pred, y).detach() # size of the output # new gradient

    optimizer.zero_grad()
    laplace_model.model.zero_grad()
    to_grad = (laplace_model.model(x) * v)
    to_grad.sum(dim=1).mean(dim=0).backward()

    lin_w_fd.grad = flatten_grad(laplace_model.model, laplace_model.batchnorm_layers)  # take the weights
    optimizer.step()

    loss_vec_fd.append(loss.detach().item())
    print(loss_vec_fd[-1])


def finite_diff_JvP(x, model, vec, eps=None):

    assert len(vec.shape) == 1
    model.eval()
    with torch.no_grad():

        batchnorm_layers = list_batchnorm_layers(model)
        map_weights = flatten_nn(model, batchnorm_layers)

        if eps is None:
            torch_eps = torch.finfo(vec.dtype).eps
            w_map_max = map_weights.abs().max().clamp(min=torch_eps)
            v_max = vec.abs().max().clamp(min=torch_eps)
            eps = np.sqrt(torch_eps) * (1 + w_map_max)  / v_max

        w_plus = map_weights.clone().detach() + vec * eps
        set_all_weights(model, batchnorm_layers, w_plus)
        f_plus = model(x)
        #         del w_plus

        w_minus = map_weights.clone().detach() - vec * eps
        set_all_weights(model, batchnorm_layers, w_minus)
        f_minus = model(x)
        #         del w_minus

        JvP = (f_plus - f_minus) / (2 * eps)
        set_all_weights(model, batchnorm_layers, map_weights)
        return JvP

def log_homoGauss_grad(mean, y, prec): # delta p(y | model) / delta mean
    return (prec * (mean - y))



# def test_optim(reconstructor, filtbackproj, store_device):
#
#     reconstructor.model.eval()
#     kappa = [[], [], [], []]
#     for lengthscale_init in np.logspace(-2, 2, 100):
#         block_priors = BlocksGPpriors(reconstructor.model, reconstructor.device, lengthscale_init)
#         lengthscales = [param for param in block_priors.parameters() if param.requires_grad]
#         expected_tv = block_priors.get_expected_TV_loss(filtbackproj)
#         dist = torch.distributions.exponential.Exponential(torch.ones(1, device=store_device))
#         for i in range(len(expected_tv)):
#             log_pi = dist.log_prob(expected_tv[i])
#             first_derivative = autograd.grad(expected_tv[i], lengthscales[i])[0] # delta_k/delta_\ell
#             log_det = first_derivative.abs().log()
#             kappa[i].append((log_pi + log_det).detach().cpu().item())
#     return kappa, np.logspace(-2, 2, 100)
