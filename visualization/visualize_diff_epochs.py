import os
import pickle
from utils import plot_template
import torch


def plot_module(PROB_NAME, DATA_NAME, excluded_keys, it_max):
    FILE_DIR = os.path.dirname(__file__)
    model_info = [PROB_NAME, DATA_NAME]
    separator = "_"
    RESULT_NAME = separator.join(model_info)

    SAVE_PATH = os.path.join(FILE_DIR, "..", "test_log", RESULT_NAME, RESULT_NAME + "_epochs.pickle")

    # Load the dictionary
    with open(SAVE_PATH, "rb") as file:
        loaded_dict = pickle.load(file)

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

    def align_length(my_dict, other_dict):
        # Iterate over the keys and remove excluded keys
        for key in my_dict:
            my_dict[key] = my_dict[key][:len(other_dict[key])]
        return my_dict

    threshold = 3e-4
    def truncate_values_vertical(my_dict, threshold, top=False):
        # Iterate over the keys and remove excluded keys
        truncated_dict = dict()
        for key in my_dict:
            lst = my_dict[key]
            if not top:
                for i,x in enumerate(lst):
                    if x < threshold:
                        lst = lst[:i]
                        break
            else:
                for i,x in enumerate(lst):
                    if x > threshold:
                        lst = lst[:i]
                        break
            truncated_dict[key] = lst
        return truncated_dict


    # def truncate_values(my_dict, bottom, top):
    #     # Iterate over the keys and remove excluded keys
    #     truncated_dict = dict()
    #     for key in my_dict:
    #         lst = my_dict[key]
    #         for i,x in enumerate(lst):
    #             if x < bottom or x > :
    #                 lst = lst[:i+1]
    #                 break
    #         truncated_dict[key] = lst
    #     return truncated_dict

    # \begin{equation}\label{eq:difference-continuous}
    #     \begin{aligned}
    #         \varphi(t)&=\frac{x(t+2h)-2 x(t+h)+x(t)}{h^2}+\frac{\alpha}{t}\frac{x(t+h)-x(t)}{h}\\
    #         &\qquad+\beta(t)\frac{\nabla f(x(t+h))-\nabla f(x(t))}{h}+\gamma(t) \nabla f(x(t)).
    #     \end{aligned}
    # \end{equation}
    # computes the quantity in the above equation
    def difference_continuous(x_continuous_history, grad_func, alpha, beta, gamma, h=0.04, t0=1.):
        difference_continuous = dict()
        for key in x_continuous_history:
            lst = x_continuous_history[key]
            grad_lst = grad_func[key]
            beta_lst = beta[key]
            gamma_lst = gamma[key]
            alpha_lst = alpha[key]
            difference_continuous[key] = []
            for i in range(len(lst)-2):
                x = lst[i]
                x1 = lst[i+1]
                x2 = lst[i+2]
                t = t0+i*h
                t1 = (i+1)*h
                t2 = (i+2)*h
                df = grad_lst[i]
                df1 = grad_lst[i+1]
                df2 = grad_lst[i+2]
                diff = (x2 - 2*x1 + x)/h**2 + alpha_lst[i]*(x1 - x)/h + beta_lst[i]*(df1 - df)/h + gamma_lst[i]*df
                difference_continuous[key].append(diff.norm().item())
        return difference_continuous

    diff_continuous = difference_continuous(loaded_dict['x_continuous_history'], loaded_dict['grad_history'], loaded_dict['alpha_history'], loaded_dict['beta_history'], loaded_dict['gamma_history'], 0.04)
    diff_continuous = truncate_values(del_keys(diff_continuous, excluded_keys), it_max)

    x_continuous_record = truncate_values(del_keys(loaded_dict['x_continuous_history'], excluded_keys), it_max)
    x_record = truncate_values(del_keys(loaded_dict['x_history'], excluded_keys), it_max)
    g_record = truncate_values(del_keys(loaded_dict['g_history'], excluded_keys), it_max)
    f_record = truncate_values(del_keys(loaded_dict['f_history'], excluded_keys), it_max)
    lambda_record = truncate_values(del_keys(loaded_dict['lambda_history'], excluded_keys), it_max)
    stable1_record = truncate_values(del_keys(loaded_dict['stable1_history'], excluded_keys), it_max)
    stable2_record = truncate_values(del_keys(loaded_dict['stable2_history'], excluded_keys), it_max)
    stable3_record = truncate_values(del_keys(loaded_dict['stable3_history'], excluded_keys), it_max)
    stable4_record = truncate_values(del_keys(loaded_dict['stable4_history'], excluded_keys), it_max)
    SAVE_DIR = os.path.join(FILE_DIR, "..", "figure_table", RESULT_NAME, "epochs")

    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    result_path = os.path.join(SAVE_DIR, RESULT_NAME)


    g_truncated = truncate_values_vertical(g_record, threshold=g_record["default"][0], top=True)
    # stable1_record = truncate_values_vertical(stable1_record, threshold=-2.)
    # stable2_record = truncate_values_vertical(stable2_record, threshold=-1.)
    # stable3_record = truncate_values_vertical(stable3_record, threshold=-1.)
    # stable4_record = truncate_values_vertical(stable4_record, threshold=-1.)
    
    stable1_record = align_length(stable1_record, g_truncated)
    stable2_record = align_length(stable2_record, g_truncated)
    stable3_record = align_length(stable3_record, g_truncated)
    stable4_record = align_length(stable4_record, g_truncated)
    
    # compute the norm of x_history minus x_continuous_history, where x_history is the discrete points and x_continuous_history is the continuous points, x_history is a dict with keys as the methods and values as the list of points, x_continuous_history is a dict with keys as the methods and values as the torch 2D tensor of points (iterate_num * var_dim)
    # I want to calculate the relative error
    x_diff = dict()
    for key in x_record:
        tensor_2d = torch.stack(x_record[key])
        x_diff[key] = torch.norm(tensor_2d - x_continuous_record[key], dim=1).tolist()
    x_diff = truncate_values(del_keys(x_diff, excluded_keys), it_max)
    
    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\Vert \varphi(t_k)\Vert$"
    upper_x = it_max
    lower_x = None
    lower_y, upper_y = threshold, None
    diff_continuous = align_length(diff_continuous, g_truncated)
    plot_template(diff_continuous, x_label, y_label, result_path+'_epochs_phi', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', set_lim=False, plot_threshold=False)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\Vert x_k-x(t_k)\Vert$"
    upper_x = it_max
    lower_x = None
    lower_y, upper_y = threshold, None
    x_diff = align_length(x_diff, g_truncated)
    plot_template(x_diff, x_label, y_label, result_path+'_epochs_diff', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', set_lim=False, plot_threshold=False)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\Vert\nabla f(x_{k})\Vert$"
    upper_x = it_max
    lower_x = None
    lower_y, upper_y = threshold, None
    plot_template(g_truncated, x_label, y_label, result_path+'_epochs_grad', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy', set_lim=False, plot_threshold=False)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\lambda_{\max}(\nabla^2 f(x_{k}))$"
    upper_x = it_max
    lower_x = 0
    lower_y, upper_y = 10**-4, 1
    # lambda_record = align_length(lambda_record, g_truncated)
    plot_template(lambda_record, x_label, y_label, result_path+'_epochs_lambda', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot')

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$f(x_{k})$"
    upper_x = it_max
    lower_x = 0
    lower_y, upper_y = 10**-4, 1
    f_record = align_length(f_record, g_truncated)
    plot_template(f_record, x_label, y_label, result_path+'_epochs_func', lower_x, upper_x, lower_y, upper_y, plot_type = 'semilogy')

    x_label = r"$\Vert\nabla f(x_{k})\Vert$"
    y_label = r"$\lambda_{\max}(\nabla^2 f(x_{k}))$"
    lower_x, upper_x = 0.001, 1.1
    lower_y, upper_y = None, None
    set_lim = False
    # lambda_record = align_length(lambda_record, g_truncated)
    plot_template(lambda_record, x_label, y_label, result_path+'_epochs_smooth', lower_x, upper_x, lower_y, upper_y, plot_type = 'loglog', line_style='', x_record=g_record, total_marker=10, set_lim=set_lim)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\rho_{k}$"
    upper_x = None
    lower_x = None
    lower_y = -.25
    upper_y = 0.25
    set_lim = True
    plot_template(stable1_record, x_label, y_label, result_path+'_epochs_stable1', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', set_lim=set_lim)

    x_label = r"$\mathrm{iteration}$"
    # y_label = r"$\frac{h^2}{4}(\beta_k\Lambda_k+\alpha/t_k)^2-h^2\gamma_k\Lambda(f,x_k)$"
    y_label = r"$C_k$"
    upper_x = None
    lower_x = None
    lower_y, upper_y = -1.1, .25
    set_lim = False
    plot_template(stable2_record, x_label, y_label, result_path+'_epochs_stable2', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', set_lim=set_lim)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\max\{1-B_k,B_k-1\}$"
    upper_x = None
    lower_x = None
    lower_y, upper_y = -1., 1.
    set_lim = False
    plot_template(stable3_record, x_label, y_label, result_path+'_epochs_stable3', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', set_lim=set_lim)

    x_label = r"$\mathrm{iteration}$"
    y_label = r"$\mathrm{stable 3}$"
    upper_x = None
    lower_x = None
    lower_y, upper_y = -1., 1.
    set_lim = False
    plot_template(stable4_record, x_label, y_label, result_path+'_epochs_stable4', lower_x, upper_x, lower_y, upper_y, plot_type = 'plot', set_lim=set_lim)

if __name__ == '__main__':
    # PROB_NAME = "logistic"
    PROB_NAME = "lpp"
    easy_cases = ["mushrooms", "a5a", "w3a", "phishing", "covtype","separable"]
    DATA_NAME = easy_cases[1]
    # excluded_keys = ["LIEIV", "DRK", "LRK", "LE2GC", "LEIGC"]
    # excluded_keys = ["INNA", "LIEIV", "DRK", "LRK", "LE2GC", "LEIGC", "IGAHD", "LFE"]
    excluded_keys = []
    it_max = 2000
    plot_module(PROB_NAME, DATA_NAME, excluded_keys, it_max)
    # for PROB_NAME in ['logistic', 'lpp']:
    #     for DATA_NAME in easy_cases:
    #         plot_module(PROB_NAME, DATA_NAME, excluded_keys, it_max)