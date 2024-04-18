import torch
from vector_field import zhangODE
from classic_optimizer import gd, nag
from .plot_template import plot_template
from .integrators import runge_kutta4, euler
import numpy as np
from .diff_eigval import Lambda
from .helper_func import del_keys, truncate_values


class test_discrete:
    def __init__(self, vf, parent_grad, parent_loss, t0, x0, it_max, d, excluded_keys = ["LIEIV", "DRK", "INNA"]) -> None:
        self.vf = vf
        self.parent_grad = parent_grad
        self.parent_loss = parent_loss
        self.grad_func = None
        self.loss_func = None
        self.result = dict()
        self.d = d
        self.t0 = t0
        self.h = 0.04
        self.x0 = x0
        self.it_max = it_max
        self.excluded_keys = excluded_keys

    def construct_prob(self, A, b):
        self.result['A'] = A
        self.result['b'] = b
        def grad_func(w): return self.parent_grad(w, A, b)
        def loss_func(w): return self.parent_loss(w, A, b)
        self.d = A.shape[1]
        self.x0 = torch.ones(self.d, device=A.device) / self.d
        self.grad_func = grad_func
        self.loss_func = loss_func
        self.vf.gradFunc = grad_func
        self.L = 2 * Lambda(grad_func, self.x0)

    def neural_vf_(self, t, y):
        x, v = self.vf.forward_no_penalty(t, (y[:self.d], y[self.d:]))
        return torch.cat((x.squeeze(), v.squeeze()))

    def zhang_vf_(self, t, y):
        x, v = zhangODE(t, (y[:self.d], y[self.d:]), p=5, grad_func=self.grad_func)
        return torch.cat((x, v))

    def test(self):
        pass

    def update_result(self):
        self.result = self.run_test()

    def run_test(self):
        t0 = self.t0
        d = self.d
        L = self.L
        h = self.h
        x0 = self.x0
        start_point = t0.item()
        grad_func = self.grad_func
        loss_func = self.loss_func
        it_max = self.it_max

        names = ["NAG", "GD", "DRK", "LEIGC", "LIEIV", "LRK", "IGAHD", "LE2GC", "LFE"]
        vfs = [nag, gd, self.zhang_vf_, self.vf, self.vf, self.neural_vf_, self.vf, self.vf, self.neural_vf_]

        result = dict()
        iterate_history = dict()
        f_history = dict()
        g_history = dict()
        lambda_history = dict()
        stable1_history = dict()
        stable2_history = dict()
        stable3_history = dict()

        interval_length = h
        num_intervals = it_max

        end_point = start_point + interval_length * num_intervals
        t = torch.arange(start_point, end_point, interval_length, device=x0.device)
        result['h'] = h
        result['t'] = t
        result['a'] = self.vf.alpha
        result['beta'] = self.vf.beta(t).squeeze().detach().cpu().numpy()
        result['gamma'] = self.vf.gamma(t).squeeze().detach().cpu().numpy()
        t = t.detach().cpu().numpy()
        for vf, name in zip(vfs, names):
            v0 = torch.zeros(d, device = x0.device)
            y0 = torch.cat((x0, v0), 0)
            f_list = np.zeros(it_max)
            g_list = np.zeros(it_max)
            lambda_list = np.zeros(it_max)
            if name == "INNA":
                y_list = euler(vf, y0, it_max, h, t0)
            elif name == "DRK":
                # y_list = euler(vf, y0, it_max, h / 10, t0)
                y_list = runge_kutta4(vf, y0, it_max, h, t0)
            elif name == "GD":
                y_list = vf(grad_func, it_max, 1 / L, x0)
            elif name == "NAG":
                y_list = vf(grad_func, it_max, 1 / L, x0)
            elif name in ["LPOLY", "POLY", "LPOLY+"]:
                y_list = euler(vf, y0, it_max, h, t0)
            elif name == "LEIGC":
                y_list = vf.eigc(x0, it_max)
            elif name == "LIEIV":
                y_list = vf.IEIV(x0, it_max)
            elif name == "LE2GC":
                y_list = vf.E2GC(x0, it_max)
            elif name == "LRK":
                v0 = x0 + self.vf.beta(t0)*grad_func(x0)
                y0 = torch.cat((x0, v0.squeeze()), dim=0)
                y_list = runge_kutta4(vf, y0, it_max, h, t0)
            elif name == "LFE":
                v0 = x0 + self.vf.beta(t0)*grad_func(x0)
                y0 = torch.cat((x0, v0.squeeze()), dim=0)
                y_list = euler(vf, y0, it_max, h, t0)
            elif name == "IGAHD":
                y_list = vf.IGAHD(x0, it_max)
            else:
                raise UserWarning("Wrong vector field name!")
            for i in range(len(y_list)):
                u = y_list[i]
                g_norm = torch.norm(grad_func(u[:d])).detach().cpu().numpy()
                g_list[i] = g_norm
                f_list[i] = loss_func(u[:d]).detach().cpu().numpy()
                lambda_list[i] = Lambda(grad_func, u[:d]).detach().cpu().numpy()
            if name in ["LEIGC", "LE2GC", "IGAHD", "LIEIV", "LRK", "LFE"]:
                stable1 = (result['beta'] - h*result['gamma']/2)*lambda_list + result['a']/t - 2/h
                stable2 = (h*result['gamma'] - result['beta'])*lambda_list - result['a']/t
                stable3 = (result['beta'] + h*result['gamma']/2)*lambda_list - result['a']/t - 2/h
                stable1_history[name] = stable1
                stable2_history[name] = stable2
                stable3_history[name] = stable3
            f_history[name] = f_list
            g_history[name] = g_list
            lambda_history[name] = lambda_list
            iterate_history[name] = y_list
        result['iterate_history'] = iterate_history
        result['f_history'] = f_history
        result['g_history'] = g_history
        result['lambda_history'] = lambda_history
        result['stable1_history'] = stable1_history
        result['stable2_history'] = stable2_history
        result['stable3_history'] = stable3_history

        return result

    def plot_grad(self):
        g_record = truncate_values(del_keys(self.result['g_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$\Vert\nabla f(x)\Vert$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(g_record, x_label, y_label, '_grad', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', save=False)
        return fig

    def plot_func(self):
        f_record = truncate_values(del_keys(self.result['f_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$f(x)$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(f_record, x_label, y_label, '_func', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', save=False)
        return fig

    def plot_stable1(self):
        stable1_record = truncate_values(del_keys(self.result['stable1_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$\mathrm{stable 1}$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(stable1_record, x_label, y_label, '_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', save=False)
        return fig

    def plot_stable2(self):
        stable2_record = truncate_values(del_keys(self.result['stable2_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$\mathrm{stable 2}$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(stable2_record, x_label, y_label, '_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', save=False)
        return fig

    def plot_stable3(self):
        stable3_record = truncate_values(del_keys(self.result['stable3_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$\mathrm{stable 3}$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(stable3_record, x_label, y_label, '_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', save=False)
        return fig

    def plot_smooth(self):
        lambda_record = truncate_values(del_keys(self.result['lambda_history'], self.excluded_keys), self.it_max)
        x_label = r"$\mathrm{iteration}$"
        y_label = r"$\lambda_{\max}(\nabla^2 f(x))$"
        upper_x = self.it_max
        lower_x = 0
        lower_y, upper_y = 10**-4, 1
        fig = plot_template(lambda_record, x_label, y_label, '_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', save=False)
        return fig

    def fetch_data(self):
        return self.result
