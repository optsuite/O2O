import os
import torch
import argparse
from utils import gen_data, train_neural_l1, init_coeffs, Lambda
from vector_field import DIN_AVD
from problem import (logistic_gradient, logistic_loss, logistic_smoothness,
                    loss_lp, grad_lp)
from datetime import datetime


def train_model(problem, dataset, pretrain, num_epoch, pen_coeff, lr=0.001, momentum=0.9, batch_size=1024, seed=None, init_it=300, h=0.04, eps=3/10**4, l2=0.0, p=4, optim="SGD", threshold=10.0):
    """
    Train a neural ODE to fit a dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:

    print(device)
    now = datetime.now()
    experiment_id = now.strftime("%Y%m%d_%H%M%S")

    model_info = [problem, dataset]
    separator = "_"
    MODEL_NAME = separator.join(model_info)
    FILE_DIR = os.path.dirname(os.path.abspath(__file__))
    print("File directory:", FILE_DIR)
    # FILE_DIR = os.path.dirname(__file__)
    # print(FILE_DIR)
    MODEL_PATH = os.path.join(FILE_DIR, "experiments/", MODEL_NAME)
    print("Model directory:", MODEL_PATH)
    SAVE_PATH = os.path.join(MODEL_PATH, str(experiment_id))
    # SAVE_PATH = os.path.join(FILE_DIR, "trained_model", MODEL_NAME + ".pth")
    print(MODEL_NAME)

    if not os.path.isdir(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    if seed:
        torch.manual_seed(seed)

    data_loader, var_dim = gen_data(
        name=dataset,
        mode="train",
        normalized=True,
        batch_size=batch_size,
        device=device,
        full_gradient=False,
    )

    test_loader, _ = gen_data(
        name=dataset,
        mode="test",
        normalized=True,
        batch_size=batch_size,
        device=device,
        full_gradient=False,
    )

    def parent_loss(w, A, b):
        if problem == "logistic":
            return logistic_loss(w, A, b, l2)
        else:
            return loss_lp(A, b, w, p)

    def parent_grad(w, A, b):
        if problem == "logistic":
            return logistic_gradient(w, A, b, l2)
        else:
            return grad_lp(A, b, w, p)

    x0 = torch.ones(var_dim, device=device) / var_dim
    # x0 = torch.zeros(var_dim, device=device)
    t0 = torch.tensor(1.0).to(device)
    vf = DIN_AVD(gradFunc=None, t0=t0, h=h, threshold=threshold).to(device)

    if pretrain:
        params = torch.load(os.path.join(FILE_DIR, 'trained_model', MODEL_NAME + ".pth"), map_location=torch.device(device))
        vf.load_state_dict(params)
    else:
        A, b = next(iter(data_loader))
        grad_func = lambda w: parent_grad(w, A, b)
        # if problem == "logistic":
        #     L = logistic_smoothness(A)
        # else:
        #     L = (p-1) * torch.max(torch.linalg.eigvalsh(A.T @ A)) * torch.max((A @ x0 - b).abs() ** (p - 2)) / A.shape[0]
        L = torch.minimum(logistic_smoothness(A), 4 * Lambda(grad_func, x0))
        _ = init_coeffs(vf, h, L, grad_func, x0, t0, init_it)

    LOG_DIR = os.path.join(FILE_DIR, "runs", MODEL_NAME)
    if optim == "SGD":
        optimizer = torch.optim.SGD(vf.parameters(), lr=lr, momentum=momentum)
    else:
        optimizer = torch.optim.Adam(vf.parameters(), lr=lr)
    train_neural_l1(
        problem,
        dataset,
        data_loader,
        device,
        parent_loss,
        parent_grad,
        vf,
        optimizer,
        pen_coeff,
        x0,
        t0,
        LOG_DIR,
        batch_size,
        num_epoch,
        eps,
        SAVE_PATH,
        test_dataloader=test_loader,
        it_max=300,
        d=var_dim
    )

    # torch.save(
    # vf.state_dict(), os.path.join(FILE_DIR, "trained_model", MODEL_NAME + ".pth")
    # )

def argument_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the neural ODE using exact L1 penalty method.",
        usage='%(prog)s [options]')
    parser.add_argument(
        "--problem",
        help="Either logistic regression (default: without L2 regularization) or Lpp minimization (default: p=4)",
        choices=["logistic", "lpp"],
        type=str)
    parser.add_argument(
        "--dataset",
        help="Dataset use for training",
        choices=["mushrooms", "a5a", "w3a", "phishing", "separable", "covtype"],
        type=str)
    parser.add_argument(
        "--pretrain",
        help="Load the pre-trained model or not",
        action='store_true'
    )
    parser.add_argument(
        "--num_epoch",
        help="The number of the training epoch",
        type=int,
        default=60
    )
    parser.add_argument(
        "--pen_coeff",
        help="The penalty coefficient of the L1 exact penalty term",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--lr",
        help="Learning rate of SGD",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--momentum",
        help="Momentum coefficient of SGD",
        type=float,
        default=0.9
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size for training, default 1024, 10240 is recommended for covtype",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--seed",
        help="Random seed for reproducing. 3407 is all you need",
        type=int,
        default=3407
    )
    parser.add_argument(
        "--init_it",
        help="The number of iterate used to initialize the neural ODE, default is 300",
        type=int,
        default=300
    )
    parser.add_argument(
        "--discrete_stepsize",
        help="the step size used in discretization, default is 0.04",
        type=float,
        default=0.04
    )
    parser.add_argument(
        "--eps",
        help="epsilon used to define the stopping time",
        type=float,
        default=1e-4
    )
    parser.add_argument(
        "--l2",
        help="the coefficient of the L2 regularization term in logistic regression",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--p",
        help="the exponential index of lpp minimization",
        type=int,
        default=4
    )
    parser.add_argument(
        "--optim",
        help="the optimizer using in training",
        type=str,
        choices=["SGD", "Adam"],
        default="SGD"
    )
    parser.add_argument(
        "--threshold",
        help="the threshold using in constraints",
        type=float,
        default=10.0
    )
    arguments = parser.parse_args()
    return arguments

if __name__ == '__main__':
    args = argument_parser()
    train_model(args.problem,
                args.dataset,
                args.pretrain,
                args.num_epoch,
                args.pen_coeff,
                args.lr,
                args.momentum,
                args.batch_size,
                args.seed,
                args.init_it,
                args.discrete_stepsize,
                args.eps,
                args.l2,
                args.p,
                args.optim)