import torch


def runge_kutta4(f, y0, it_max, h, t0):
    n = it_max
    y = torch.zeros((n, len(y0)), device=y0.device)
    y[0] = y0
    safe_guard = 5 * f(t0, y0).norm()
    for i in torch.arange(n - 1):
        t = t0 + i * h
        k1 = f(t, y[i])
        k2 = f(t + h / 2.0, y[i] + k1 * h / 2.0)
        k3 = f(t + h / 2.0, y[i] + k2 * h / 2.0)
        k4 = f(t + h, y[i] + k3 * h)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if k1.norm() > safe_guard:
            break
    return y.to(y0.device)


def euler(f, y0, it_max, h, t0):
    n = it_max
    y = torch.zeros((n, len(y0)), device=y0.device)
    y[0] = y0
    safe_guard = 5 * f(t0, y0).norm()
    for i in torch.arange(n - 1):
        t = t0 + i * h
        k1 = f(t, y[i])
        if k1.norm() > safe_guard:
            break
        y[i + 1] = y[i] + h * k1
    return y.to(y0.device)