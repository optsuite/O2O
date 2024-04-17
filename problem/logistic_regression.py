import torch
import torch.linalg as la
from torch.special import expit


def logistic_smoothness(X):
    return torch.max(torch.linalg.eigvalsh(X.T @ X / X.shape[0]))


class LogisticGradient:
    def __init__(self, X=None, y=None, l2=0., normalize=True):
        self.X = X
        self.y = y
        self.l2 = l2
        self.normalize = normalize

    def forward(self, w):
        return logistic_gradient(w, self.X, self.y, self.l2, self.normalize)

    def set_params(self, X, y, l2, normalize=True):
        self.X = X
        self.y = y
        self.l2 = l2
        self.normalize = normalize

    def __call__(self, w):
        return self.forward(w)

from memory_profiler import profile

# @profile
def logistic_gradient(w, X, y_, l2, normalize=True):
    """
    Gradient of the logistic loss at point w with features X, labels y and l2 regularization.
    If labels are from {-1, 1}, they will be changed to {0, 1} internally
    """
    y = (y_+1) / 2 if -1 in y_ else y_
    activation = torch.special.expit(X @ w)
    grad = X.T@(activation - y) / X.shape[0] + l2 * w
    if normalize:
        return grad
    return grad * len(y)


def logistic_loss(w, X, y, l2):
    """Logistic loss, numerically stable implementation.

    Parameters
    ----------
    w: array-like, shape (n_features,)
        Coefficients

    X: array-like, shape (n_samples, n_features)
        Data matrix

    y: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    loss: float
    """
    z = X @ w
    # y = y.view(len(y), 1)
    return torch.mean((1 - y) * z - logsig(z)) + l2 / 2 * la.norm(w) ** 2


def logistic_hess(w, X, l2):
    z = X @ w
    s = expit(z)
    return s * (1 - s) * X.T @ X / X.shape[0] + l2


def logsig(x):
    """
    Compute the log-sigmoid function component-wise.
    See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.
    """
    out = torch.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - torch.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -torch.log1p(torch.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -torch.exp(-x[idx3])
    return out