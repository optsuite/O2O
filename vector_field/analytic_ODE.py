import torch
import torch.nn as nn


class DIN(nn.Module):
    def __init__(self, d_f=None, alpha=1.0, beta=0.5):
        super(DIN, self).__init__()
        self.grad_func = d_f
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.name = 'DIN'

    def forward(self, t, y):
        x, v = y
        dvdt = (1 / self.beta - self.alpha) * x - v / self.beta
        dxdt = dvdt - self.beta * self.grad_func(x)
        return dxdt, dvdt

    def cond(self, t, y, hess_func):
        x, v = y
        var_dim = len(x)
        H = hess_func(x)
        I = torch.eye(var_dim)
        hess = torch.zeros([2 * var_dim, 2 * var_dim])
        hess[:var_dim, :var_dim] = (1 / self.beta - self.alpha) * I - self.beta * H
        hess[:var_dim, var_dim:] = -I / self.beta
        hess[var_dim:, :var_dim] = (1 / self.beta - self.alpha) * I
        hess[var_dim:, var_dim:] = -I / self.beta
        return torch.linalg.cond(hess)

    def opt_forward(self, y):
        x, v = torch.chunk(y, 2)
        d_v = (1 / self.beta - self.alpha) * x - v / self.beta
        d_x = d_v - self.beta * self.grad_func(x)
        return torch.cat((d_x, d_v), dim=0)

    def step(self, x, v, lr):
        common_update = (1 / self.beta - self.alpha) * x - v / self.beta
        x = x + lr * (common_update - self.beta * self.grad_func(x))
        v = v + lr * common_update
        return x, v

    def run(self, x0, v0, lr, it_max):
        x_list = [x0]
        x, v = x0, v0
        for i in range(it_max):
            x, v = self.step(x, v, lr)
            x_list.append(x)
        return x_list


def zhangODE(t, y, p, grad_func):
    x, v = y
    dv = - (2*p+1) / t * v - p ** 2 * t ** (p - 2) * grad_func(x)
    return v, dv


def suODE(t, y, grad_func):
    x, v = y
    d_x = v
    d_v = -3 * v / t - grad_func(x)
    return d_x, d_v
