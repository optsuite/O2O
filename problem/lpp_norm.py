import torch

def grad_lp(A, b, x, p=4):
    pred = A@x - b
    return A.T @ torch.pow(pred, p-1) / A.shape[0]


def loss_lp(A, b, x, p=4):
    pred = A@x - b
    return torch.sum(torch.pow(pred, p)) / p / A.shape[0]