# TrainNeural
# This program aims to train an NeuralODE using sample-based SQP with inexact ADMM.
import os
from tqdm import tqdm
import torch
from torchdiffeq import odeint, odeint_event
from utils.helperFunc import plot_constraint_violate, plot_lambda, plot_coeff
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from utils.helperFunc import flatten_params, unflatten_params, form_polyhedron, admm_nn
from utils.test_discrete import test_discrete
# import psutil


mse_loss = torch.nn.MSELoss()
# from pyinstrument import Profiler
# HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
# HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

# METRIC_ACCURACY = 'accuracy'

from memory_profiler import profile

# FILE_DIR = os.path.dirname(__file__)
def train_neural_l1(PROB_NAME, DATA_NAME, data_loader, device, parentLoss, parentGradient, vf, optimizer, pen_coeff, x0, t0, log_dir, batch_size, num_epoch, eps, SAVE_PATH, test_dataloader = None, it_max = 300, d = None):

    tester = test_discrete(vf=vf, parent_grad=parentGradient, parent_loss=parentLoss, t0=t0, x0=x0, it_max=it_max, d=d)

    template = 'Iter:{}  Stop Time:{}'
    writer = SummaryWriter(SAVE_PATH + '/log/')
    exp, ssi, sei = hparams(hparam_dict={
                                        'lr' : optimizer.param_groups[0]['lr'],
                                        'momentum' : optimizer.param_groups[0]['momentum'],
                                        'batch_size' : batch_size,
                                        'pen_coeff' : pen_coeff,
                                        'problem' : PROB_NAME,
                                        'dataset' : DATA_NAME
                                        },
                            metric_dict={
                                        'hparam/stop_time' : 0,
                                        'hparam/loss' : 0,
                                        'hparam/stable_pen2' : 0
                                        })
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    if test_dataloader:
        A, b = next(iter(test_dataloader))
        A = A.to_dense()
        A = A.to(device)
        b = b.to(device)
        tester.construct_prob(A, b)
        tester.update_result()
        writer.add_figure('test-stable 1', tester.plot_stable1(), 0)
        writer.add_figure('test-stable 2', tester.plot_stable2(), 0)
        writer.add_figure('test-stable 3', tester.plot_stable3(), 0)
        writer.add_figure('test-grad', tester.plot_grad(), 0)
        writer.add_figure('test-func', tester.plot_func(), 0)
        writer.add_figure('test-smooth', tester.plot_smooth(), 0)
    iter_num = 0
    for epoch in range(num_epoch):
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        pbar.set_description(f'Epoch [{epoch + 1}/{num_epoch}]')
        for i, (A, b) in pbar:
            if len(b) < batch_size:
                break
            iter_num += 1
            stopTime, loss, t, Lambda_, beta_penalty, gamma_penalty, conv_pen1, conv_pen2, stable_pen1, stable_pen2, beta_ind, gamma_ind, stable_ind1, stable_ind2, conv_ind1, conv_ind2 = inner_train_loop(A, b, device, parentGradient, vf, optimizer, pen_coeff, x0, t0, eps)
            writer.add_scalar('stopping time', stopTime, iter_num)
            writer.add_scalar('beta_penalty', beta_penalty, iter_num)
            writer.add_scalar('gamma_penalty', gamma_penalty, iter_num)
            writer.add_scalar('conv_pen1', conv_pen1, iter_num)
            writer.add_scalar('conv_pen2', conv_pen2, iter_num)
            writer.add_scalar('stable_pen1', stable_pen1, iter_num)
            writer.add_scalar('stable_pen2', stable_pen2, iter_num)
            writer.add_scalar('loss', loss, iter_num)

            writer.add_figure('Lambda vs. steps', plot_lambda(t, Lambda_), iter_num)
            writer.add_figure('gamma vs. steps', plot_lambda(t, -gamma_ind), iter_num)
            writer.add_figure('beta vs. steps', plot_lambda(t, -beta_ind), iter_num)

            writer.add_figure('stable indicator 1 vs. steps', plot_lambda(t, -stable_ind1), iter_num)
            writer.add_figure('stable indicator 2 vs. steps', plot_lambda(t, -stable_ind2), iter_num)
            writer.add_figure('conv indicator 1 vs. steps', plot_lambda(t, -conv_ind1), iter_num)
            writer.add_figure('conv indicator 2 vs. steps', plot_lambda(t, -conv_ind2), iter_num)

            writer.add_scalar('hparam/stop_time', stopTime, iter_num)
            writer.add_scalar('hparam/loss', loss, iter_num)
            writer.add_scalar('hparam/stable_pen2', stable_pen2, iter_num)
            pbar.set_postfix({'Stopping Time' : stopTime.item(), 'loss' : loss.item()}, refresh=True)
            # print(template.format(iter_num, stopTime.item()))

        for name, param in vf.named_parameters():
            writer.add_histogram(name, param, epoch)

        checkpoint = {
        'epoch': epoch,
        'model_state_dict': vf.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
        }

        if test_dataloader:
            A, b = next(iter(test_dataloader))
            A = A.to_dense()
            A = A.to(device)
            b = b.to(device)
            tester.construct_prob(A, b)
            tester.update_result()
            writer.add_figure('test-stable 1', tester.plot_stable1(), epoch+1)
            writer.add_figure('test-stable 2', tester.plot_stable2(), epoch+1)
            writer.add_figure('test-stable 3', tester.plot_stable3(), epoch+1)
            writer.add_figure('test-grad', tester.plot_grad(), epoch+1)
            writer.add_figure('test-func', tester.plot_func(), epoch+1)
            writer.add_figure('test-smooth', tester.plot_smooth(), epoch+1)
            checkpoint['test_data'] = tester.fetch_data()

        if not os.path.isdir(SAVE_PATH + '/checkpoints'):
            os.mkdir(SAVE_PATH + '/checkpoints')
        torch.save(checkpoint, SAVE_PATH + '/checkpoints/epoch_%s.pth' % (str(epoch+1)))
        torch.save(vf.state_dict(), SAVE_PATH + '/trained_model.pth')

    writer.close()

@profile
def inner_train_loop(A, b, device, parentGradient, vf, optimizer, pen_coeff, x0, t0, eps):
    # Get the virtual memory usage
    # vmem = psutil.virtual_memory()

    # Print the total, used, and available memory
    # print(f"Total Memory: {vmem.total / (1024 * 1024):.2f} MiB")

    A = A.to_dense()
    A = A.to(device)
    b = b.to(device)
    def grad_func(w): return parentGradient(w, A, b)

    def event_fn(t, y):
        x, v = y[0], y[1]
        nrmG = torch.norm(grad_func(x))
        crit = torch.log10(nrmG) - torch.log10(torch.ones(1, device=device) * eps)
        return crit

    vf.gradFunc=grad_func

    v0 = x0 + vf.beta(t0)*vf.gradFunc(x0)
    vf.refresh()

    p0 = torch.zeros([], device=x0.device)
    # options = dict(step_size=vf.h)
    # stopTime, solution = odeint_event(vf, (x0, v0) + (p0,) * 6, t0, event_fn=event_fn, options=options, reverse_time=False, odeint_interface=odeint, method='euler', atol=1e-5, rtol=1e-4)
    # print(f"Used Memory: {vmem.used / (1024 * 1024):.2f} MiB")
    # print(f"Available Memory: {vmem.available / (1024 * 1024):.2f} MiB")

    # Get the percentage of used memory
    # print(f"Memory Usage Percentage: {vmem.percent}%")
    stopTime, solution = odeint_event(vf, (x0, v0) + (p0,) * 6, t0, event_fn=event_fn, reverse_time=False, odeint_interface=odeint)

    beta_penalty = solution[2][-1]
    gamma_penalty = solution[3][-1]
    stable_pen1 = solution[4][-1]
    stable_pen2 = solution[5][-1]
    conv_pen1 = solution[6][-1]
    conv_pen2 = solution[7][-1]

    loss = stopTime + pen_coeff * (beta_penalty + gamma_penalty + stable_pen1 + stable_pen2 + conv_pen1 + conv_pen2)

    Lambda_ = torch.tensor(vf.Lambda, device=x0.device).view(-1,1)
    t = torch.tensor(vf.time, device=x0.device).view(-1,1)

    beta_ind = -vf.beta(t).squeeze()
    gamma_ind = -vf.gamma(t).squeeze()
    stable_ind1 = stability_indicator_1(Lambda_, vf.alpha, vf.beta, vf.gamma, vf.h, t).squeeze()
    stable_ind2 = stability_indicator_2(Lambda_, vf.alpha, vf.beta, vf.gamma, vf.h, t).squeeze()
    conv_ind1 = converge_indicator_1(vf.beta, vf.gamma, t).squeeze()
    conv_ind2 = converge_indicator_2(vf.alpha, vf.beta, vf.gamma, vf.h, t).squeeze()

    # .backward() will call forward() once, hence make the length of vf.Lambda does not match t
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    vf.refresh()

    return stopTime, loss, t, Lambda_, beta_penalty, gamma_penalty, conv_pen1, conv_pen2, stable_pen1, stable_pen2, beta_ind, gamma_ind, stable_ind1, stable_ind2, conv_ind1, conv_ind2


def random_average_project(vf, proj_step_size, stability_indicator):
    alpha, beta, gamma, record_Lambda, t0, h = vf.alpha, vf.beta, vf.gamma, vf.Lambda, vf.t0, vf.h
    K = len(record_Lambda)
    act_cons = 0

    mean_project = []
    indicator_list = []
    for param in vf.parameters():
        mean_project.append(torch.zeros_like(param))
    for k in range(K):
        tk = t0 + k*h
        Lambda = record_Lambda[k]
        indicator = stability_indicator(Lambda, alpha, beta, gamma, h, tk)
        indicator_list.append(indicator.data)
        if indicator > 0:
            act_cons += 1
            vf.zero_grad()
            grads = torch.autograd.grad(indicator, vf.parameters(), create_graph=True)
            norm_square = F_norm_square(grads)
            for idx, gi in enumerate(grads):
                mean_project[idx] += indicator.item()*gi/norm_square
    if act_cons > 0:
        for idx, mean_proj in enumerate(mean_project):
            mean_project[idx] /= act_cons
        for param, mean_proj in zip(vf.parameters(), mean_project):
                param.data.copy_(param.data - proj_step_size*mean_proj)
    else:
        pass
    return indicator_list



def calc_average_project(vf, indicator_list):
    act_cons = 0
    mean_project = []
    for param in vf.parameters():
        mean_project.append(torch.zeros_like(param))
    for indicator in indicator_list:
        if indicator > 0:
            act_cons += 1
            vf.zero_grad()
            grads = torch.autograd.grad(indicator, vf.parameters(), create_graph=True)
            norm_square = F_norm_square(grads)
            for idx, gi in enumerate(grads):
                mean_project[idx] += indicator.item()*gi/norm_square
    if act_cons > 0:
        for idx, _ in enumerate(mean_project):
            mean_project[idx] /= act_cons
    return tuple(mean_project)

def zeros_like_params(params):
    zeros = []
    for param in params:
        zeros.append(torch.zeros_like(param))
    return tuple(zeros)

def calc_max_project(vf, indicator_list):
    indicator = torch.max(indicator_list)
    max_project = []
    for param in vf.parameters():
        max_project.append(torch.zeros_like(param))
    if indicator > 0:
        vf.zero_grad()
        grads = torch.autograd.grad(indicator, vf.parameters(), create_graph=True)
        norm_square = F_norm_square(grads)
        for idx, gi in enumerate(grads):
            max_project[idx] += indicator.item()*gi/norm_square
    return tuple(max_project)

def update_params(params, updates, step_size):
    for param, update in zip(params, updates):
        param.data.copy_(param.data + step_size*update)

def calc_indicator(t0, h, alpha, beta, gamma, record_Lambda, stability_indicator):
    K = len(record_Lambda)
    indicator_list = []
    for k in range(K):
        tk = t0 + k*h
        Lambda = record_Lambda[k]
        indicator = stability_indicator(Lambda, alpha, beta, gamma, h, tk)
        indicator_list.append(indicator.data)
    return indicator_list

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

def F_norm_square(grads):
    norm_square = 0.
    for grad in grads:
        norm_square += grad.norm()**2
    if len(grads) == 0:
        norm_square = 1.
    return norm_square