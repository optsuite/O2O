import torch


def gd(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0), device=x0.device)
    x = torch.clone(x0)
    for k in range(it_max):
        x_list[k] = x
        gradient = - df(x)
        x += h * gradient
    return x_list.to(x0.device)


def nag(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0), device=x0.device)
    x = torch.clone(x0)
    x_prev = torch.clone(x0)
    y = torch.clone(x0)
    for k in range(it_max):
        x_list[k] = x
        x, x_prev = y - h*h * df(y), x
        y = x + (k-1.0)/(k+2.0)*(x-x_prev)
    return x_list.to(x0.device)

def eigc(df, it_max, h, x0):
    h = h
    x, x_old = x0.clone(), x0.clone()
    v = -h * df(x)
    grad = df(x)
    x_list = torch.zeros(it_max, len(x0), device=x0.device)
    for k in range(it_max):
        x_list[k] = x
        alpha = 1. + 3./(k+1.)
        beta = h
        gamma = 1. + 3./(k+1.)

        grad_old = grad.clone()
        x = x + h * v
        grad = df(x)

        v = v - gamma*h*grad - beta*(grad - grad_old)
        v /= alpha
        # x_list.append(x)
    return x_list

def igahd(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0), device=x0.device)
    x = torch.clone(x0)
    x_old = torch.clone(x0)
    y = torch.clone(x0)
    grad = df(x)
    grad_old = grad.clone()
    for k in range(it_max):
        x_list[k] = x
        alpha = 1. - 3./(k+1.)
        beta = 1.9 * h
        gamma = 1. + 1./(k+1.)

        y = x + alpha * (x - x_old) - beta*h*(grad - grad_old)-beta*h*(gamma - 1.0)*grad_old
        x_old = x.clone()
        x = y - h*h*df(y)

        grad_old = grad.clone()
        grad = df(x)
    return x_list.to(x0.device)


def rgd(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0))
    x = torch.clone(x0)
    for k in range(it_max):
        x_list[k] = x
        gradient = - df(x)
        scale = torch.pow(torch.linalg.norm(gradient),2/3)
        x += h * gradient/scale
    return x_list


def argd(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0))
    x = torch.clone(x0)
    x_prev = torch.clone(x0)
    y = torch.clone(x0)
    for k in range(it_max):
        x_list[k] = x
        x, x_prev = y - h * df(y)/(torch.pow(torch.linalg.norm(df(y))+1e-5,2/3)), x
        y = x + (k-1)/(k+4.0)*(x-x_prev) - h**3*(k+1)*(k+2)*(k+3)*df(y)/(torch.pow(torch.linalg.norm(df(y))+1e-5,2/3))
    return x_list


def fista(df, it_max, h, x0):
    x_list = torch.zeros(it_max, len(x0), device=x0.device)
    x = torch.clone(x0)
    x_prev = torch.clone(x0)
    t = 1.0
    for k in range(it_max):
        x_list[k] = x
        y = x + h * df(x)
        x_prev, x = x, y + ((t - 1.0) / (t + 2.0)) * (y - x_prev)
        t = (1.0 + (1.0 + 4.0 * t**2)**0.5) / 2.0
    return x_list.to(x0.device)

