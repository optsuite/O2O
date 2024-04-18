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
from torchdiffeq import odeint


def compare_epochs(PROB_NAME, DATA_NAME):
    result_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    RESULT_NAME = separator.join(result_info)

    batch_size = 1024 if DATA_NAME != "covtype" else 10240
    prob = probGenerator(PROB_NAME, DATA_NAME, batch_size, mode="test")
    prob.generate()
    x0 = torch.ones(prob.var_dim) / prob.var_dim

    L = torch.minimum(logistic_smoothness(prob.A), 4 * Lambda(prob.grad_func, x0))
    for i in range(3):
        x0 = x0 - prob.grad_func(x0)/L
    t0 = torch.ones([])
    vf = DIN_AVD(gradFunc=prob.grad_func, t0=t0, h=0.04)

    # FILE_DIR = os.path.abspath(os.path.join('.'))
    FILE_DIR = os.path.dirname(__file__)
    model_info = [PROB_NAME, DATA_NAME]
    separator = '_'
    MODEL_NAME = separator.join(model_info)
    experiment_id = experiment_dict[MODEL_NAME]

    it_max = 2000
    result = dict()
    x_continuous_history = dict()
    alpha_history = dict()
    beta_history = dict()
    gamma_history = dict()
    x_history = dict()
    f_history = dict()
    g_history = dict()
    grad_history = dict()
    lambda_history = dict()
    stable1_history = dict()
    stable2_history = dict()
    stable3_history = dict()
    stable4_history = dict()
    g_list = np.zeros(it_max)
    grad_list = np.zeros((it_max, prob.var_dim))
    f_list = np.zeros(it_max)
    lambda_list = np.zeros(it_max)
    cons1 = np.zeros(it_max)
    cons2 = np.zeros(it_max)
    cons3 = np.zeros(it_max)
    it = np.arange(it_max)

    compared_epochs = [0, 10, -1, -2]
    labels = ["initial", "epoch 10", "epoch 80", "relay"]
    for num_epochs, label in zip(compared_epochs, labels):
        if num_epochs == 0:
            init_coeffs(vf=vf, h=0.04, L=L, grad_func=prob.grad_func, x0=x0, t0=t0)
            y_list = vf.EIND_untrun(x0, it_max)
            # y_list = vf.eigc(x0, it_max)
        elif num_epochs == -1:
            params = torch.load(os.path.join(FILE_DIR, "..", "train_log", MODEL_NAME, experiment_id, 'trained_model.pth'), map_location="cpu")
            vf.load_state_dict(params)
            y_list = vf.EIND_untrun(x0, it_max)
        elif num_epochs == -2:
            params = torch.load(os.path.join(FILE_DIR, "..", "train_log", MODEL_NAME, experiment_id, 'trained_model.pth'), map_location="cpu")
            vf.load_state_dict(params)
            y_list = vf.EIND(x0, it_max)
        else:
            checkpoint = torch.load(os.path.join(FILE_DIR, "..", "train_log", MODEL_NAME, experiment_id, 'checkpoints', 'epoch_' + str(num_epochs) + '.pth'), map_location="cpu")
            params = checkpoint['model_state_dict']
            vf.load_state_dict(params)
            y_list = vf.EIND_untrun(x0, it_max)
        # evolve the system using odeint for all points in t_span
        t_span = torch.linspace(t0.item(), t0.item() + (it_max-1) * vf.h, it_max)
        v0 = x0 + vf.beta(t0) * prob.grad_func(x0)
        X, V = odeint(vf, (x0, v0), t_span)
        x_continuous_history[label] = X
        x_history[label] = y_list
        for i in range(len(y_list)):
            u = y_list[i]
            x = u[:prob.var_dim]
            t = t0 + i*vf.h
            grad_list[i] = vf.gradFunc(x).detach().numpy()
            g_norm = torch.norm(vf.gradFunc(x)).detach().numpy()
            g_list[i] = g_norm
            f_list[i] = prob.loss_func(x).detach().numpy()
            lambda_list[i] = Lambda(vf.gradFunc, x).detach().numpy()
        alpha_list, beta_list, gamma_list = vf.return_coeffs()
        cons1, cons2, cons3, cons4 = admit_rho_list(torch.Tensor(lambda_list).squeeze(), vf.h, alpha_list, beta_list, gamma_list)
        f_history[label] = f_list.copy()
        g_history[label] = g_list.copy()
        alpha_history[label] = alpha_list
        beta_history[label] = beta_list
        gamma_history[label] = gamma_list
        grad_history[label] = grad_list.copy()
        lambda_history[label] = lambda_list.copy()
        stable1_history[label] = cons1.copy()
        stable2_history[label] = cons2.copy()
        stable3_history[label] = cons3.copy()
        stable4_history[label] = cons4.copy()

    result['t_span'] = t_span
    result['alpha_history'] = alpha_history
    result['beta_history'] = beta_history
    result['gamma_history'] = gamma_history
    result['x_continuous_history'] = x_continuous_history
    result['x_history'] = x_history
    result['f_history'] = f_history
    result['g_history'] = g_history
    result['grad_history'] = grad_history
    result['lambda_history'] = lambda_history
    result['stable1_history'] = stable1_history
    result['stable2_history'] = stable2_history
    result['stable3_history'] = stable3_history
    result['stable4_history'] = stable4_history
    SAVE_PATH = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME)
    result['save_path'] = SAVE_PATH
    # Store the dictionary
    with open(SAVE_PATH + "_epochs.pickle", "wb") as file:
        pickle.dump(result, file)


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

def admit_rho_list(lambda_list, h, alpha, beta, gamma):
    A = (beta*lambda_list + alpha) * h / 2.
    B = gamma * lambda_list * h * h
    C = torch.maximum(A*A - B, torch.zeros_like(A))
    D = torch.maximum(1 - 2*A + B, torch.zeros_like(A))
    E = A*A - B
    F = torch.minimum(2. - B, B)
    G = 1 - 2*A + B

    constraint1 = 1. - D**0.5
    constraint2l = 2. - B - C**0.5
    constraint3l = B - C**0.5

    # Initialize constraint_l with the same size as constraint1
    constraint_l = torch.zeros_like(constraint1)

    # When E >= 0
    mask_geq_0 = E >= 0
    constraint_l[mask_geq_0] = torch.minimum(constraint1[mask_geq_0], 
                                            torch.minimum(constraint2l[mask_geq_0], 
                                                        constraint3l[mask_geq_0]))

    # When E < 0
    mask_lt_0 = E < 0
    constraint_l[mask_lt_0] = constraint1[mask_lt_0]
    return constraint_l.detach().numpy(), E.detach().numpy(), F.detach().numpy(), G.detach().numpy()


if __name__ == "__main__":
    prob_list = ["logistic", "lpp"]
    data_list = ["mushrooms", "a5a", "w3a", "phishing", "covtype", "separable"]
    prob_id = 1
    data_id = 3
    PROB_NAME = prob_list[prob_id]
    DATA_NAME = data_list[data_id]
    compare_epochs(PROB_NAME, DATA_NAME)