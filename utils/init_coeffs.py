import torch

def init_coeffs(vf, h, L, grad_func, x0, t0, it_max = 400):
    it_max = 1200

    sqrt_s = torch.sqrt(1 / L) * torch.ones(it_max, device=x0.device)
    t = torch.linspace(-it_max*h, h, it_max, device=x0.device)

    fit_curve(vf.beta, t, sqrt_s.pow(2) / h)
    fit_curve(vf.gamma, t, sqrt_s.pow(2) / h**2)

    t = torch.linspace(h, it_max*h, it_max, device=x0.device)
    beta_bound = (2./h - 1. * vf.alpha/t) * sqrt_s.pow(2)
    gamma_bound = beta_bound / h
    # beta_bound = sqrt_s.pow(2) / h
    # gamma_bound = sqrt_s.pow(2) * (1 + vf.alpha * h / t / 2.) / h**2

    fit_curve(vf.beta, t, beta_bound)
    fit_curve(vf.gamma, t, gamma_bound)


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