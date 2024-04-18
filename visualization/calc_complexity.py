import os
import sys
import torch
import numpy as np
from utils import probGenerator, Lambda, init_coeffs, plot_template
from problem import logistic_smoothness
from classic_optimizer import gd, nag, eigc, igahd
from vector_field import DIN, DIN_AVD, zhangODE
import matplotlib.pyplot as plt
import pickle


def cal_complexity(PROB_NAME, DATA_NAME):
    result_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    RESULT_NAME = separator.join(result_info)

    batch_size = 1024 if DATA_NAME != "covtype" else 10240
    prob = probGenerator(PROB_NAME, DATA_NAME, batch_size, mode="test")
    prob.generate()
    x0 = torch.ones(prob.var_dim) / prob.var_dim

    L = torch.minimum(logistic_smoothness(prob.A), 4 * Lambda(prob.grad_func, x0))
    for i in range(1):
        x0 = x0 - prob.grad_func(x0)/L
    t0 = torch.ones([])

    experiment_dict = dict()
    experiment_dict['logistic_a5a'] = '20230620_021417'
    experiment_dict['logistic_mushrooms'] = '20230620_021422'
    experiment_dict['logistic_w3a'] = '20230620_021417'
    experiment_dict['logistic_covtype'] = '20230621_101012'
    experiment_dict['logistic_separable'] = '20230620_021417'
    experiment_dict['logistic_phishing'] = '20230620_021417'
    experiment_dict['lpp_a5a'] = '20230620_220134'
    experiment_dict['lpp_mushrooms'] = '20230620_021417'
    experiment_dict['lpp_w3a'] = '20230620_083009'
    experiment_dict['lpp_covtype'] = '20230621_025737'
    experiment_dict['lpp_separable'] = '20230621_021909'
    experiment_dict['lpp_phishing'] = '20230620_021417'

    def cal_single_complexity(y_list, prob, threshold):
        grad_func = prob.grad_func
        it_max = len(y_list)
        g_list = np.zeros(it_max)
        for i in range(it_max):
            u = y_list[i]
            x = u[:prob.var_dim]
            g_norm = torch.norm(grad_func(x)).detach().numpy()
            g_list[i] = g_norm
        complexity = np.where(g_list < threshold)[0]
        if len(complexity) > 0:
            complexity = complexity[0]
        else:
            complexity = -1
        return complexity
            
    def runge_kutta4(f, y0, it_max, h, t0):
        n = it_max
        y = torch.zeros((n, len(y0)))
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
        return y


    def euler(f, y0, it_max, h, t0):
        n = it_max
        y = torch.zeros((n, len(y0)))
        y[0] = y0
        safe_guard = 5 * f(t0, y0).norm()
        for i in torch.arange(n - 1):
            t = t0 + i * h
            k1 = f(t, y[i])
            if k1.norm() > safe_guard:
                break
            y[i + 1] = y[i] + h * k1
        return y

    def gen_iterate(name, vf, prob, it_max):
        h = 0.04
        grad_func = prob.grad_func
        d = prob.var_dim
        v0 = torch.zeros(d)
        y0 = torch.cat((x0, v0), 0)
        if name == "INNA":
            y_list = euler(vf, y0, it_max, h, t0)
        elif name == "DRK":
            # y_list = euler(vf, y0, it_max, h / 10, t0)
            y_list = runge_kutta4(vf, y0, it_max, h, t0)
        elif name == "GD":
            y_list = vf(grad_func, it_max, 1 / L, x0)
        elif name == "NAG":
            y_list = vf(grad_func, it_max, 1 / L, x0)
        elif name == "IGAHD":
            y_list = vf.EIND(x0, it_max)
        elif name == "EIGC":
            y_list = vf.eigc(x0, it_max)
        elif name in ["LPOLY", "POLY", "LPOLY+"]:
            y_list = euler(vf, y0, it_max, h, t0)
        elif name == "LEIGC":
            y_list = vf.eigc(x0, it_max)
        elif name == "LIEIV":
            y_list = vf.IEIV(x0, it_max)
        elif name == "LE2GC":
            y_list = vf.E2GC(x0, it_max)
        elif name == "INVD(learned)":
            y_list = vf.EIND(x0, it_max)
        elif name == "INVD(initial)":
            y_list = vf.EIND_untrun(x0, it_max)
        else:
            raise UserWarning("Wrong vector field name!")
        return y_list

    FILE_DIR = os.path.dirname(__file__)
    experiment_id = experiment_dict[RESULT_NAME]
    params = torch.load(
    os.path.join(FILE_DIR, "..", "train_log", RESULT_NAME, experiment_id, 'trained_model.pth'), map_location="cpu"
    )
    it_max = 500
    h = 0.04
    d = prob.var_dim
    neural_vf = DIN_AVD(t0=t0, h=h, gradFunc=prob.grad_func, record=False)
    neural_vf.load_state_dict(params)
    neural_vf_init = DIN_AVD(t0=t0, h=h, gradFunc=prob.grad_func, record=False)
    init_coeffs(neural_vf_init, h, L, prob.grad_func, x0, t0, it_max = 400)

    def neural_vf_(t, y):
        x, v = neural_vf(t, (y[:d], y[d:]))
        return torch.cat((x.squeeze(), v.squeeze()))

    names = ["NAG", "GD", "INVD(initial)", "EIGC", "INVD(learned)"]
    vfs = [nag, gd, neural_vf_init, neural_vf_init, neural_vf]

    num_tests = 2
    threshold = 3e-4

    complexity_record = dict()
    for name in names:
        complexity_record[name] = np.zeros(num_tests)
    for i in range(num_tests):
        prob.generate()
        for (name, vf) in zip(names, vfs):
            y_list = gen_iterate(name, vf, prob, it_max)
            complexity = cal_single_complexity(y_list, prob, threshold)
            complexity_record[name][i] = complexity
    SAVE_DIR = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    SAVE_PATH = os.path.join(SAVE_DIR, RESULT_NAME)
    # Store the dictionary
    with open(SAVE_PATH + "_complexity.pickle", "wb") as file:
        pickle.dump(complexity_record, file)

if __name__ == "__main__":
    prob_list = ["logistic", "lpp"]
    data_list = ["mushrooms", "a5a", "w3a", "phishing", "covtype", "separable"]
    # prob_id = 1
    # data_id = 2
    for prob_id in range(2):
        for data_id in range(6):
            PROB_NAME = prob_list[prob_id]
            DATA_NAME = data_list[data_id]
            cal_complexity(PROB_NAME, DATA_NAME)