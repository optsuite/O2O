import torch
from .data_preprocess import gen_data
from problem import logistic_loss, logistic_gradient, grad_lp, loss_lp


class probGenerator():
    def __init__(self, prob_name, data_name, batch_size, mode) -> None:
        self.prob_name = prob_name
        self.data_name = data_name
        self.batch_size = batch_size
        self.mode = mode
        self.data_loader, self.var_dim = gen_data(
            name=data_name,
            mode=mode,
            normalized=True,
            full_gradient=False,
            batch_size=batch_size)

    def generate(self):
        A, self.b = next(iter(self.data_loader))
        self.A = A.to_dense()

    def loss_func(self, w):
        if self.prob_name == "logistic":
            return logistic_loss(w, self.A, self.b, l2=0.)
        else:
            return loss_lp(self.A, self.b, w, p=4)

    def grad_func(self, w):
        if self.prob_name == "logistic":
            return logistic_gradient(w, self.A, self.b, l2=0.)
        else:
            return grad_lp(self.A, self.b, w, p=4)