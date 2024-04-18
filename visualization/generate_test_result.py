import sys
import os

# Add the project's root directory to the module search path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
from utils import gen_data, Lambda, init_coeffs
from classic_optimizer import gd, nag, eigc, igahd
import matplotlib.pyplot as plt
from vector_field import DIN, DIN_AVD, zhangODE
from problem import (
    logistic_loss,
    logistic_smoothness,
    logistic_gradient,
    grad_lp,
    loss_lp
)
import time
import pickle
import numpy as np


def test_module(PROB_NAME, DATA_NAME, FILE_DIR, experiment_id, batch_size):
    result = dict()
    model_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    MODEL_NAME = separator.join(model_info)

    result_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    RESULT_NAME = separator.join(result_info)

    result['problem'] = PROB_NAME
    result['dataset'] = DATA_NAME

    # seed = 3407
    # torch.manual_seed(seed)

    h = 0.04
    full_gradient = False
    test_loader, var_dim = gen_data(
        name=DATA_NAME,
        mode="test",
        normalized=True,
        full_gradient=full_gradient,
        batch_size=batch_size,
    )
    A, b = next(iter(test_loader))
    A = A.to_dense()
    n, d = A.shape
    l2 = 0
    p = 4

    result["A"] = A
    result["b"] = b

    def loss_func(w):
        if PROB_NAME == "logistic":
            return logistic_loss(w, A, b, l2)
        else:
            return loss_lp(A, b, w, p)

    def grad_func(w):
        if PROB_NAME == "logistic":
            return logistic_gradient(w, A, b, l2)
        else:
            return grad_lp(A, b, w, p)

    # x0 = torch.ones(var_dim) / var_dim
    x0 = torch.zeros(var_dim)

    L = torch.minimum(logistic_smoothness(A), 4 * Lambda(grad_func, x0))

    # x0 = torch.matmul(V, torch.mul(S_pinv, torch.matmul(U.t(), b)))
    for i in range(3):
        x0 = x0 - grad_func(x0) / L
    # AtA_diag = torch.einsum('ij,ij->j', [A, A])
    # AtA_diag_ = torch.diagonal(A @ A.T)
    # x0 = (b @ A) @ (A.T @ A + 1e-3 * torch.eye(var_dim)).inverse()
    # x0 = (A.T @ A + 10.0 * torch.eye(var_dim)).inverse() @ (b @ A)

    FILE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(FILE_DIR, "..", "trained_model", "./")

    DIN_vf = DIN(d_f=grad_func)


    def DIN_vf_(t, y):
        x, v = DIN_vf(t, (y[:d], y[d:]))
        return torch.cat((x, v))

    t0 = torch.tensor(1.0)
    neural_vf = DIN_AVD(t0=t0, h=h, gradFunc=grad_func, record=False)

    # if MODEL_NAME == 'logistic_covtype':
    #     checkpoint = torch.load(os.path.join(FILE_DIR, "..", "experiments", MODEL_NAME, experiment_id, 'checkpoints', 'epoch_10.pth'), map_location="cpu")
    #     params = checkpoint['model_state_dict']
    # else:
    #     params = torch.load(
    #         os.path.join(FILE_DIR, "..", "experiments", MODEL_NAME, experiment_id, 'trained_model.pth'), map_location="cpu"
    #     )
    params = torch.load(
    os.path.join(FILE_DIR, "..", "train_log", MODEL_NAME, experiment_id, 'trained_model.pth'), map_location="cpu"
    )
    it_max = 300
    neural_vf.load_state_dict(params)
    neural_vf_init = DIN_AVD(t0=t0, h=h, gradFunc=grad_func, record=False)
    init_coeffs(neural_vf_init, h, L, grad_func, x0, t0, it_max = 400)

    def neural_vf_(t, y):
        x, v = neural_vf(t, (y[:d], y[d:]))
        return torch.cat((x.squeeze(), v.squeeze()))


    def zhang_vf_(t, y):
        x, v = zhangODE(t, (y[:d], y[d:]), p=5, grad_func=grad_func)
        return torch.cat((x, v))


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

    names = ["NAG", "GD", "EIGC", "IGAHD", "DRK", "INNA", "LEIGC", "LIEIV", "LRK", "INVD", "LE2GC", "LFE"]
    vfs = [nag, gd, neural_vf_init, neural_vf_init, zhang_vf_, DIN_vf_, neural_vf, neural_vf, neural_vf_, neural_vf, neural_vf, neural_vf_]

    iterate_history = dict()
    f_history = dict()
    g_history = dict()
    lambda_history = dict()
    stable1_history = dict()
    stable2_history = dict()
    stable3_history = dict()


    start_point = t0.item()
    interval_length = h
    num_intervals = it_max

    end_point = start_point + interval_length * num_intervals
    t = torch.arange(start_point, end_point, interval_length)
    result['h'] = h
    result['t'] = t
    result['a'] = neural_vf.alpha
    result['beta'] = neural_vf.beta(t).squeeze().detach().numpy()
    result['gamma'] = neural_vf.gamma(t).squeeze().detach().numpy()
    t = t.detach().numpy()
    for vf, name in zip(vfs, names):
        v0 = torch.zeros(d)
        y0 = torch.cat((x0, v0), 0)
        f_list = np.zeros(it_max)
        g_list = np.zeros(it_max)
        lambda_list = np.zeros(it_max)
        stable1_list = np.zeros(it_max)
        stable2_list = np.zeros(it_max)
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
        elif name == "LRK":
            v0 = x0 + neural_vf.beta(t0)*grad_func(x0)
            y0 = torch.cat((x0, v0.squeeze()), dim=0)
            y_list = runge_kutta4(vf, y0, it_max, h, t0)
        elif name == "LFE":
            v0 = x0 + neural_vf.beta(t0)*grad_func(x0)
            y0 = torch.cat((x0, v0.squeeze()), dim=0)
            y_list = euler(vf, y0, it_max, h, t0)
        elif name == "INVD":
            y_list = vf.EIND(x0, it_max)
        else:
            raise UserWarning("Wrong vector field name!")
        for i in range(len(y_list)):
            u = y_list[i]
            g_norm = torch.norm(grad_func(u[:d])).detach().numpy()
            g_list[i] = g_norm
            f_list[i] = loss_func(u[:d]).detach().numpy()
            lambda_list[i] = Lambda(grad_func, u[:d]).detach().numpy()
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
    SAVE_DIR = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    SAVE_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
    result['save_path'] = SAVE_PATH
    # Store the dictionary
    with open(SAVE_PATH + ".pickle", "wb") as file:
        pickle.dump(result, file)


if __name__ == '__main__':

    easy_cases = ["mushrooms", "a5a", "w3a", "phishing", "covtype", "separable"]
    # named_tuple = time.localtime()
    # TIME_STR = time.strftime("%m%d_%H%M", named_tuple)
    FILE_DIR = os.path.dirname(__file__)
    separator = "_"

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

    # PROB_NAME = "logistic"
    PROB_NAME = "lpp"
    DATA_NAME = easy_cases[4]
    model_info = [PROB_NAME, DATA_NAME]
    MODEL_NAME = separator.join(model_info)
    if DATA_NAME == 'covtype':
        batch_size = 10240
    else:
        batch_size = 1024
    # test_module(PROB_NAME, DATA_NAME, FILE_DIR, experiment_id=experiment_dict[MODEL_NAME], batch_size=batch_size)

    for PROB_NAME in ['logistic', 'lpp']:
        for DATA_NAME in easy_cases:
            model_info = [PROB_NAME, DATA_NAME]
            MODEL_NAME = separator.join(model_info)
            if DATA_NAME == 'covtype':
                batch_size = 10240
            else:
                batch_size = 1024
            test_module(PROB_NAME, DATA_NAME, FILE_DIR, experiment_id=experiment_dict[MODEL_NAME], batch_size=batch_size)