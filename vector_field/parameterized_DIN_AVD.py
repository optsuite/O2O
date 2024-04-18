import torch
import torch.nn as nn
from torch.autograd.functional import jvp
from math import sqrt, log
from utils import power_iteration, Lambda


class DIN_AVD(nn.Module):
    def __init__(self, gradFunc, t0, h, record=True):
        super().__init__()
        self.gradFunc = gradFunc
        self.alpha = 6

        hidden_size = 25.
        init_len = sqrt(2.)/sqrt(hidden_size)
        self.beta = NetWithDeriv(a=init_len/2, b=init_len*3/2, val=-init_len)
        self.gamma = NetWithDeriv(a=init_len/2, b=init_len*30/2, val=-init_len)
        self.name = 'neuralODE'

        self.t0 = t0
        self.h = h
        self.idx = 0
        self.Lambda = []
        self.record = record
        self.alpha_list = []
        self.beta_list = []
        self.gamma_list = []

    def forward(self, t, y):
        x, v = y
        t = t.view(1)

        # if self.record:
        #     self.record_Lambda(t, x)
        eigMax = power_iteration(self.gradFunc, x)
        self.Lambda.append(eigMax)


        # hessian = torch.autograd.functional.jacobian(self.gradFunc, x)
        # eigMax = torch.linalg.eigvals(hessian)[0].norm()
        # self.Lambda.append(eigMax.data)

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
        self.idx = 0

    def eigc(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        safe_guard = 3e-4
        x_list = []
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 + self.alpha/tk*h
            beta = self.beta(tk).item()
            # beta = 3.0
            gamma = self.gamma(tk).item()
            df_old = df.clone()

            x = x + h * v
            df = self.gradFunc(x)
            # if df.norm() > safe_guard:
            #     break

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
        safe_guard = 3e-4
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            # beta = 3.0
            gamma = self.gamma(tk).item()
            # df_old = df.clone()
            df = self.gradFunc(x + beta/gamma * v)

            v = alpha*v - h*gamma*df

            x = x + h * v
            x_list.append(x)
        return x_list

    def E2GC(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        x_list = []
        safe_guard = 3e-4
        for k in range(1, it_max+1):
            tk = t0 + k*h
            # tk = k*h*torch.ones([])
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            # beta = 3.0
            gamma = self.gamma(tk).item()
            df_old = df.clone()

            x = x + h * v
            df = self.gradFunc(x)

            v = alpha*v - gamma*h*df - beta*(df - df_old)
            x_list.append(x)
        return x_list

    def EIND(self, x0, it_max):
        t0 = self.t0
        tk = t0.clone()
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        df_old = df.clone()
        x_list = []
        safe_guard = 3e-4
        truncate = False
        for k in range(1, it_max+1):
            x_list.append(x)
            tk = tk + h

            # if df.norm() > safe_guard and truncate:
            alpha = 1 - self.alpha/tk*h

            if df.norm() > df_old.norm() and df_old.norm() < 3e-4 and not truncate:
                truncate = True
                lambda_trun = Lambda(self.gradFunc, x)
                # x_old = x.clone()
                # df_old = df.clone()

                # x = x - df / lambda_trun
                # df = self.gradFunc(x)
                # v = torch.zeros_like(x0)
                # tk = t0.clone()
                # lambda_trun = (df - df_old).norm()/(x - x_old).norm()
            if truncate:
                beta = (2./h - 1. * self.alpha/tk) / lambda_trun
                gamma = beta / h
                # v = torch.zeros_like(x0)
                truncate = True
            else:
                beta = self.beta(tk).item()
                gamma = self.gamma(tk).item()
            # gamma = self.gamma(tk).item()

            # y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*(gamma - 1.0)*df_old
            y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*gamma*df_old
            x_old = x.clone()
            x = y.clone()
            # x = y - h*h*self.gradFunc(y)

            df_old = df.clone()
            df = self.gradFunc(x)
            self.alpha_list.append(self.alpha/tk)
            self.beta_list.append(beta)
            self.gamma_list.append(gamma)
            # print(self.alpha_list)
        return x_list

    def EIND_untrun(self, x0, it_max):
        t0 = self.t0
        h = self.h
        x, x_old = x0.clone(), x0.clone()
        v = torch.zeros_like(x0)
        df = self.gradFunc(x)
        df_old = df.clone()
        x_list = []
        for k in range(1, it_max+1):
            x_list.append(x)
            tk = t0 + k*h
            alpha = 1 - self.alpha/tk*h
            beta = self.beta(tk).item()
            gamma = self.gamma(tk).item()
            y = x + alpha * (x - x_old) - beta*h*(df - df_old)-h*h*gamma*df_old
            x_old = x.clone()
            x = y.clone()

            df_old = df.clone()
            df = self.gradFunc(x)
            self.alpha_list.append(self.alpha/tk)
            self.beta_list.append(beta)
            self.gamma_list.append(gamma)
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

    def return_coeffs(self):
        alpha_list = torch.Tensor(self.alpha_list).clone()
        beta_list = torch.Tensor(self.beta_list).clone()
        gamma_list = torch.Tensor(self.gamma_list).clone()
        self.alpha_list = []
        self.beta_list = []
        self.gamma_list = []
        return alpha_list, beta_list, gamma_list


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
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.constant_(m.bias, val=val)
        #         nn.init.uniform_(m.weight, a=a, b=b)

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