import sys
import os

# Add the project's root directory to the module search path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import pickle
import matplotlib.pyplot as plt
from problem import (
    logistic_loss,
    logistic_gradient,
    grad_lp,
    loss_lp
)
from utils import Lambda, plot_template


def plot_module(PROB_NAME, DATA_NAME, excluded_keys, it_max):
    FILE_DIR = os.path.dirname(__file__)
    model_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    RESULT_NAME = separator.join(model_info)

    # SAVE_DIR = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME)

    SAVE_PATH = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME, RESULT_NAME + ".pickle")

    # Load the dictionary
    with open(SAVE_PATH, "rb") as file:
        loaded_dict = pickle.load(file)

    A = loaded_dict['A']
    b = loaded_dict['b']
    l2 = 0.0
    p = 4

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

    def del_keys(my_dict, excluded_keys):
        # Iterate over the keys and remove excluded keys
        for key in excluded_keys:
            if key in my_dict:
                del my_dict[key]
        return my_dict

    def truncate_values(my_dict, it_max):
        # Iterate over the keys and remove excluded keys
        for key in my_dict:
            my_dict[key] = my_dict[key][:it_max]
        return my_dict

    threshold = 3e-4
    def truncate_values_vertical(my_dict, threshold):
        # Iterate over the keys and remove excluded keys
        truncated_dict = dict()
        for key in my_dict:
            lst = my_dict[key]
            for i,x in enumerate(lst):
                if x < threshold:
                    lst = lst[:i+1]
                    break
            truncated_dict[key] = lst
        return truncated_dict

    def relative_error(my_dict):
        # Iterate over the absolute error and calculate the relative error
        relative_error = dict()
        min_val = 10.
        for key in my_dict:
            min_val = min(min(my_dict[key]), min_val)
        for key in my_dict:
            relative_error[key] = (my_dict[key] - min_val) / min_val + 1e-8
        return relative_error

    x_record = truncate_values(del_keys(loaded_dict['iterate_history'], excluded_keys), it_max)
    g_record = truncate_values(del_keys(loaded_dict['g_history'], excluded_keys), it_max)
    g_truncated = truncate_values_vertical(g_record, threshold)
    f_record = truncate_values(del_keys(loaded_dict['f_history'], excluded_keys), it_max)
    lambda_record = truncate_values(del_keys(loaded_dict['lambda_history'], excluded_keys), it_max)
    stable1_record = truncate_values(del_keys(loaded_dict['stable1_history'], excluded_keys), it_max)
    stable2_record = truncate_values(del_keys(loaded_dict['stable2_history'], excluded_keys), it_max)
    stable3_record = truncate_values(del_keys(loaded_dict['stable3_history'], excluded_keys), it_max)
    SAVE_DIR = os.path.join(FILE_DIR, "..", "figure_table", RESULT_NAME)

    if not os.path.isdir(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    result_path = os.path.join(SAVE_DIR, RESULT_NAME)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\Vert\nabla f(x)\Vert$"
    upper_x = it_max
    lower_x = None
    lower_y, upper_y = threshold, None
    plot_template(g_truncated, x_label, y_label, result_path+'_grad', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', set_lim=False, plot_threshold=True)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\lambda_{\max}(\nabla^2 f(x))$"
    upper_x = it_max
    lower_x = 0
    lower_y, upper_y = 10**-4, 1
    plot_template(lambda_record, x_label, y_label, result_path+'_lambda', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy')

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$f(x)$"
    upper_x = it_max
    lower_x = 0
    lower_y, upper_y = 10**-4, 1
    plot_template(f_record, x_label, y_label, result_path+'_func', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy')

    x_label = r"$\Vert\nabla f(x)\Vert$"
    y_label = r"$\lambda_{\max}(\nabla^2 f(x))$"
    lower_x, upper_x = 0.001, 1.1
    lower_y, upper_y = None, None
    set_lim = False
    plot_template(lambda_record, x_label, y_label, result_path+'_smooth', lower_x, upper_x, lower_y, upper_y, plot_type = 'loglog', line_style='', x_record=g_record, total_marker=10, set_lim=set_lim)

    # x_label = r"$\mathrm{iteration}$"
    # y_label = r"$\mathrm{stable 1}$"
    # upper_x = it_max
    # lower_x = 0
    # lower_y, upper_y = 10**-4, 1
    # plot_template(stable1_record, x_label, y_label, result_path+'_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot')

    # x_label = r"$\mathrm{iteration}$"
    # y_label = r"$\mathrm{stable 2}$"
    # upper_x = it_max
    # lower_x = 0
    # lower_y, upper_y = 10**-4, 1
    # plot_template(stable2_record, x_label, y_label, result_path+'_stable2', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot')

    # x_label = r"$\mathrm{iteration}$"
    # y_label = r"$\mathrm{stable 3}$"
    # upper_x = it_max
    # lower_x = 0
    # lower_y, upper_y = 10**-4, 1
    # plot_template(stable3_record, x_label, y_label, result_path+'_stable3', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot')


if __name__ == '__main__':
    # PROB_NAME = "logistic"
    PROB_NAME = "lpp"
    easy_cases = ["mushrooms", "a5a", "w3a", "phishing", "covtype", "separable"]
    DATA_NAME = easy_cases[4]
    # excluded_keys = ["LIEIV", "DRK", "LRK", "LE2GC", "LEIGC"]
    excluded_keys = ["IGAHD", "INNA", "LIEIV", "DRK", "LRK", "LE2GC", "LEIGC", "LFE"]
    # excluded_keys = []
    it_max = 300
    for PROB_NAME in ["lpp", "logistic"]:
        for DATA_NAME in easy_cases:
            plot_module(PROB_NAME, DATA_NAME, excluded_keys, it_max)