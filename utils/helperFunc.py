import torch
import matplotlib.pyplot as plt
import numpy as np


def del_keys(my_dict, excluded_keys):
    # Iterate over the keys and remove excluded keys
    for key in excluded_keys:
        if key in my_dict:
            del my_dict[key]
    return my_dict

def truncate_values(my_dict, it_max):
    # Iterate over the keys and remove excluded keys
    for key in my_dict:
        my_dict[key] = my_dict[key][:it_max]
    return my_dict

def plot_Fobj(Fobj_list):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.plot([indicator.cpu().detach().numpy() for indicator in Fobj_list], label='Function Value')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel(r'$k$', fontsize=20)
    ax.set_ylabel(r'$f(x_k)$', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(axis='both', color='.95', linestyle='-', linewidth=2)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    return fig

def plot_Gnrm(Gnrm_list):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    ax.semilogy([indicator.cpu().detach().numpy() for indicator in Gnrm_list], label='Gradient Norm')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel(r'$k$', fontsize=20)
    ax.set_ylabel(r'$\Vert g(x_k)\Vert$', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(axis='both', color='.95', linestyle='-', linewidth=2)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    return fig

def plot_constraint_violate(t0, h, indicator_list):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    K = len(indicator_list)
    T = t0 + (K-1)*h
    t = np.linspace(t0.cpu().detach().numpy(), T.cpu().detach().numpy(), K, endpoint=True)

    ax.plot(t, [indicator.item() for indicator in indicator_list], label='Indicator Violate')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel(r'$t_k$', fontsize=20)
    # ax.set_ylabel(r'$\alpha/t_k+2/h-\beta(t_k)-h\gamma(t_k)\Lambda(f,x_k)/2$', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(axis='both', color='.95', linestyle='-', linewidth=2)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    return fig

def plot_lambda(t_list, lambda_list):
    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    K = len(lambda_list)
    # T = t0 + (K-1)*h
    # t = np.linspace(t0.cpu().detach().numpy(), T.cpu().detach().numpy(), K, endpoint=True)

    ax.plot([t.item() for t in t_list], [eigMax.item() for eigMax in lambda_list], label='Largest Eigenvalue of Hessian')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel(r'$t_k$', fontsize=20)
    ax.set_ylabel(r'$\mathrm{indicator}$', fontsize=20)
    # ax.legend(fontsize=20)
    ax.grid(axis='both', color='.95', linestyle='-', linewidth=2)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    return fig


def plot_coeff(t0, vf, K, type):
    h = vf.h
    if type == 'beta':
        name = r'$\beta(tk)$'
        coeff = vf.beta
    elif type == 'gamma':
        name = r'$\gamma(tk)$'
        coeff = vf.gamma

    fig = plt.figure(figsize=(6, 4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)

    T = t0 + (K-1)*h
    t = np.linspace(t0.cpu().detach().numpy(), T.cpu().detach().numpy(), K, endpoint=True)
    coeff_list = [coeff(t0 + k*h) for k in range(K)]

    ax.plot(t, [coeff_val.item() for coeff_val in coeff_list], label='Coefficient value')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=18)
    ax.set_xlabel(r'$t_k$', fontsize=20)
    ax.set_ylabel(name, fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(axis='both', color='.95', linestyle='-', linewidth=2)
    fig.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)

    return fig

def generate_Lambda(grad_func, x_list):
    Lambda_list = []
    for x in x_list:
        hessian = torch.autograd.functional.jacobian(grad_func, x)
        eigMax = torch.linalg.eigvals(hessian)[0].norm()
        # eigMax, _ = torch.lobpcg(hessian)
        Lambda_list.append(eigMax.data)
    return Lambda_list

def generate_Fobj(loss_func, x_list):
    Fobj_list = []
    for x in x_list:
        Fobj_list.append(loss_func(x).data)
    return Fobj_list

def generate_Gnrm(grad_func, x_list):
    Gnrm_list = []
    for x in x_list:
        Gnrm_list.append(grad_func(x).norm().data)
    return Gnrm_list

def fit_curve(model, x, y):
    # Define the loss function
    loss_fn = torch.nn.MSELoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train the model
    for i in range(2000):
        y_pred = model(x.unsqueeze(1))
        loss = loss_fn(y_pred, y.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# def flatten_params(model):
#     """
#     Flatten the model's parameters (weights and biases) or the gradient 
#     of the parameters into a single vector.
#     """
#     return torch.cat([param.view(-1) for param in model.parameters() if param.requires_grad])

def unflatten_params(model, flat_params):
    """
    Update the model's parameters with the provided flat_params vector.
    """
    idx = 0
    for param in model.parameters():
        param_len = param.numel()
        param.data.copy_(flat_params[idx:idx+param_len].view(param.shape))
        idx += param_len

def flatten_params(params):
    """
    Flatten the model's parameters (weights and biases) or the gradient 
    of the parameters into a single vector.
    """
    return torch.cat([param.view(-1) for param in params])

def form_polyhedron(vf, indicator_list):
    """
    Form the linear inequality constraints for sample-based QP problem.
    """

    W = flatten_params(vf.parameters())
    m = len(indicator_list)
    n = W.numel()

    C = torch.zeros([m, n], device=W.device)
    d = torch.zeros(m, device=W.device)
    torch.autograd.set_detect_anomaly(True)

    for i in range(m):
        indicator = indicator_list[i]
        vf.zero_grad()
        grads = torch.autograd.grad(indicator, vf.parameters(), retain_graph=True)

        C[i] = flatten_params(grads)
        d[i] = C[i].dot(W) - indicator.data
    vf.zero_grad(set_to_none=True)
    return C, d

def admm_nn(alpha, q, C, d, rho, x_init, max_iter=1000, tol=1e-6):
    device = q.device
    n = x_init.shape[0]
    m = C.shape[0]

    # Initialize variables
    x = x_init.clone()
    x_i = torch.zeros((m, n), device=device)
    norm_x_init = torch.norm(x_init, p=2)
    lambda_i = torch.zeros((m, n), device=device)

    for k in range(max_iter):
        # Update x
        x_prev = x.clone()
        x = (rho * torch.sum(x_i - lambda_i / rho, dim=0) - q) / (1 / alpha + m * rho)

        # Update x_i using parallel projection
        for i in range(m):
            z = x + lambda_i[i] / rho
            norm_c_i = torch.norm(C[i], p=2)
            if C[i].dot(z) > d[i]:
                x_i[i] = z - C[i] * (C[i].dot(z) - d[i]) / norm_c_i**2
            else:
                x_i[i] = z

        # Update lambda_i
        if m > 0:
            lambda_i += rho * (x - x_i)

        # Check for convergence
        if torch.norm(x - x_prev) / norm_x_init < tol:
            break

    return x

def power_iteration(grad_func, x, num_iterations=10):
    # Initialize a random vector
    v = torch.randn_like(x)
    x_ = x.detach().requires_grad_(True)
    
    for _ in range(num_iterations):
        # Compute the gradient of the function at x
        # grad = torch.autograd.grad(func(x), x, create_graph=True)[0]

        # Compute the Jacobian
        # Compute the Jacobian-vector product (JVP) of the gradient with respect to v
        # Hv = torch.autograd.grad(grad, x, v, retain_graph=True)[0]
        # Hv = torch.func.jvp(grad_func, x, v)[1]
        # Hv = torch.autograd.functional.jvp(grad_func, x, v)[1]
        with torch.enable_grad():
            Hv, = torch.autograd.grad(grad_func(x_), x_, v, retain_graph=True, allow_unused=True)
        #     Hv, = torch.func.jvp()
        # Hv = torch.func.jvp(grad_func, (x_,), (v,))[1]

        # Update the estimate of the largest eigenvector
        v = Hv / torch.norm(Hv)

        # Compute the largest eigenvalue using the Rayleigh quotient
    # largest_eigenvalue = (v @ Hv).item()
    largest_eigenvalue = v @ Hv

    return largest_eigenvalue