import torch
import torch.nn as nn
from torch.autograd.functional import jvp
from math import sqrt, log
from utils import power_iteration, Lambda

m = nn.ReLU()


class DIN_AVD_augmented(nn.Module):
    def __init__(self, gradFunc, t0, h, record=True, threshold=10.0):
        super().__init__()
        self.gradFunc = gradFunc
        self.alpha = 6

        hidden_size = 25.
        init_len = sqrt(2.)/sqrt(hidden_size)
        self.beta = NetWithDeriv(a=init_len/2, b=init_len*3/2, val=-init_len)
        self.gamma = NetWithDeriv(a=init_len/2, b=init_len*30/2, val=-init_len)
        self.name = 'DIN_AVD_augmented'

        self.t0 = t0
        self.h = h
        self.idx = 0
        self.Lambda = []
        self.time = []
        self.record = record
        self.threshold = 10.0
    # @profile
    def forward(self, t, y):
        x, v = y[0], y[1]
        t = t.view(1)

        alpha = self.alpha
        beta = self.beta(t)
        gamma = self.gamma(t)
        dbeta = self.beta.deriv(t)
        df = self.gradFunc(x)
        dx = -x + v - beta * df
        dv = (1 - alpha / t) * dx + (dbeta - gamma) * df

        Lambda_ = Lambda(self.gradFunc, x)
        # Lambda_ = torch.zeros([])
        self.Lambda.append(Lambda_)
        self.time.append(t)
        beta_ind = -self.beta(t).squeeze()
        gamma_ind = -self.gamma(t).squeeze()
        stable_ind1 = stability_indicator_1(Lambda_, self.alpha, self.beta, self.gamma, self.h, t).squeeze()
        stable_ind2 = stability_indicator_2(Lambda_, self.alpha, self.beta, self.gamma, self.h, t).squeeze()
        conv_ind1 = converge_indicator_1(self.beta, self.gamma, t).squeeze()
        conv_ind2 = converge_indicator_2(self.alpha, self.beta, self.gamma, self.h, t).squeeze()

        return dx, dv, torch.maximum(beta_ind, torch.zeros([])), torch.maximum(gamma_ind, torch.zeros([])), torch.maximum(stable_ind1 + self.threshold/t, torch.zeros([])), torch.maximum(stable_ind2 + self.threshold/t, torch.zeros([])), torch.maximum(conv_ind1, torch.zeros([])), torch.maximum(conv_ind2, torch.zeros([]))
    
    def forward_no_penalty(self, t, y):
        x, v = y[0], y[1]
        t = t.view(1)

        alpha = self.alpha
        beta = self.beta(t)
        gamma = self.gamma(t)
        dbeta = self.beta.deriv(t)
        df = self.gradFunc(x)
        dx = -x + v - beta * df
        dv = (1 - alpha / t) * dx + (dbeta - gamma) * df

        return dx, dv

    def record_Lambda(self, t, x):
        if self.h is not None and self.idx != torch.div(t - self.t0, self.h, rounding_mode='floor'):
            self.idx += 1
            # hessian = torch.autograd.functional.jacobian(self.gradFunc, x)
            # eigMax = torch.linalg.eigvals(hessian)[0].norm()
            # # eigMax, _ = torch.lobpcg(hessian)
            # self.Lambda.append(eigMax.data)
            eigMax = power_iteration(self.gradFunc, x)
            self.Lambda.append(eigMax)
        else:
            pass

    def refresh(self):
        self.Lambda = []
        self.time = []
        self.idx = 0

    def eigc(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        safe_guard = 5 * torch.norm(df)
        x_list = []
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 + self.alpha/tk*h
            beta = self.beta(tk).item()
            gamma = self.gamma(tk).item()
            df_old = df.clone()

            x = x + h * v
            df = self.gradFunc(x)
            if df.norm() > safe_guard:
                break

            v = v - gamma*h*df - beta*(df - df_old)
            v /= alpha
            x_list.append(x)
        return x_list

    def IEIV(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        v_old = v.clone()
        df = self.gradFunc(x)
        x_list = []
        safe_guard = 5 * torch.norm(df)
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            gamma = self.gamma(tk).item()
            # df_old = df.clone()
            df = self.gradFunc(x + beta/gamma * v)

            v = alpha*v - h*gamma*df

            x = x + h * v

            if df.norm() > safe_guard:
                break
            # v = v - gamma*h*df - beta*(df - df_old)
            # v /= alpha
            x_list.append(x)
        return x_list

    def E2GC(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        x_list = []
        safe_guard = 5 * torch.norm(df)
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            gamma = self.gamma(tk).item()
            df_old = df.clone()

            x = x + h * v
            df = self.gradFunc(x)
            if df.norm() > safe_guard:
                break

            v = alpha*v - gamma*h*df - beta*(df - df_old)
            x_list.append(x)
        return x_list

    def EIND(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        df_old = df.clone()
        x_list = []
        safe_guard = 5 * df.norm()
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            # beta = 3.0
            gamma = self.gamma(tk).item()

            # y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*(gamma - 1.0)*df_old
            y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*gamma*df_old
            x_old = x.clone()
            x = y.clone()
            # x = y - h*h*self.gradFunc(y)

            df_old = df.clone()
            df = self.gradFunc(x)

            if df.norm() > safe_guard:
                break

            x_list.append(x)
        return x_list

    def IGAHD(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        df_old = df.clone()
        x_list = []
        safe_guard = 3e-4
        truncate = True
        for k in range(1, it_max+1):
            x_list.append(x)
            tk = t0 + k*h

            # if df.norm() > safe_guard and truncate:
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            gamma = self.gamma(tk).item()
            truncate = False

            y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*(gamma - 1.0)*df_old
            # y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*gamma*df_old
            x_old = x.clone()
            # x = y.clone()
            x = y - h*h*self.gradFunc(y)

            df_old = df.clone()
            df = self.gradFunc(x)
        return x_list

class NetWithDeriv(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, a=0.0, b=1.0, val=0.):
        super(NetWithDeriv, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(hidden_size, output_size)

        self.grad1 = lambda x: torch.minimum(torch.exp(x), torch.ones_like(x))
        self.grad2 = lambda x: torch.minimum(torch.exp(x), torch.ones_like(x))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.zeros_(m.bias)
                torch.nn.init.xavier_normal_(m.weight)

    def forward(self, t):
        out = self.fc1(t.view(-1, 1))
        out = self.elu1(out)
        out = self.fc2(out)
        out = self.elu2(out)
        out = self.fc3(out)
        return out

    def deriv(self, t):
        out = self.fc1(t.view(-1, 1)).T
        deriv = self.grad1(out)
        deriv = torch.mul(self.fc1.weight, deriv)
        deriv = torch.matmul(self.fc2.weight, deriv)
        out = self.elu1(out)
        out = self.fc2(out.T)
        deriv = torch.mul(self.grad2(out.T), deriv)
        deriv = torch.matmul(self.fc3.weight, deriv).squeeze()
        return deriv

    def derivative(self, t):
        return self.deriv(t)

    def evaluate(self, t):
        return self(t)

def stability_indicator_1(Lambda, alpha, beta, gamma, h, t):
    return (beta(t) - h*gamma(t)/2)*Lambda + alpha/t - 2/h

def stability_indicator_2(Lambda, alpha, beta, gamma, h, t):
    return (h*gamma(t) - beta(t))*Lambda - alpha/t

def converge_indicator_1(beta, gamma, t):
    return beta(t)/t + beta.deriv(t).view(-1, 1) - gamma(t)

def converge_indicator_2(alpha, beta, gamma, diff_length, t):
    w = -converge_indicator_1(beta, gamma, t)
    w_ = -converge_indicator_1(beta, gamma, t+diff_length)
    approx_deriv = (w_ - w)/diff_length
    return t*approx_deriv - (alpha-3)*w