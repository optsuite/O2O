import os
import torch
import numpy as np
from math import floor
import torch.utils.data as data
import scipy.linalg as LA
import scipy
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer


scaler = MaxAbsScaler()
# scaler = Normalizer()
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset', './')


def gen_data(name='mushrooms', batch_size=128, mode='train', device=torch.device("cpu"), normalized=True, full_gradient=False):
    if mode == 'train':
        if name != 'separable':
            raw_data = load_svmlight_file(DATA_DIR + name)
        if name == 'mushrooms':
            n_train = 4096
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
            y -= 1
        elif name == 'news20':
            n_train = 4996
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
        elif name == 'protein':
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
            y -= 1
        elif name == 'covtype':
            n_train = 102400
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
            y -= 1
        elif name == 'phishing':
            n_train = 8192
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
        elif name == 'madelon':
            n_train = 2000
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
        elif name == 'connect-4':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 10240
            X, y = raw_data[0][:n_train], raw_data[1][:n_train]
        elif name=='separable':
            # generate linearly separable dataset
            # this code is directly adapted from ARGD: https://github.com/aswilson07/ARGD
            np.random.seed(12)
            num_observations = 10240
            clusters = 1
            dim = 100
            spread = 20
            margin = np.random.randint(10, size=dim) /10

            for i in range(clusters):
                if i == 0:
                    x1 = np.random.multivariate_normal(np.random.randint(spread, size=dim) + margin, np.identity(dim), num_observations)
                    x2 = np.random.multivariate_normal(np.random.randint(spread, size=dim), np.identity(dim), num_observations)
                else:
                    x1 = np.append(x1, np.random.multivariate_normal(np.random.randint(spread, size=dim) + margin, np.identity(dim), num_observations), axis=0)
                    x2 = np.append(x2, np.random.multivariate_normal(np.random.randint(spread, size=dim), np.identity(dim), num_observations), axis=0)
            X = np.zeros((2 * num_observations, dim))
            simulated_labels = np.zeros(X.shape[0])
            X[list(range(0, 2*num_observations, 2)),:] = x1
            X[list(range(1, 2*num_observations+1, 2)),:] = x2
            simulated_labels[list(range(0, 2*num_observations, 2))] = 1
            intercept = np.ones((X.shape[0], 1))
            simulated_separableish_features = np.hstack((intercept, X))
            dim += 1
            X = simulated_separableish_features
            y = simulated_labels
        else:
            X, y = raw_data[0], raw_data[1]
    elif mode == 'test':
        if name == 'mushrooms':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 4096
            X, y = raw_data[0][n_train:], raw_data[1][n_train:]
            y -= 1
        elif name == 'news20':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 4996
            X, y = raw_data[0][n_train:], raw_data[1][n_train:]
        elif name == 'covtype':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 102400
            X, y = raw_data[0][n_train: 2 * n_train], raw_data[1][n_train: 2 * n_train]
            y -= 1
        elif name == 'phishing':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 8192
            X, y = raw_data[0][n_train:], raw_data[1][n_train:]
        elif name == 'connect-4':
            raw_data = load_svmlight_file(DATA_DIR + name)
            n_train = 10240
            X, y = raw_data[0][n_train: 2*n_train], raw_data[1][n_train: 2*n_train]
        elif name=='separable':
            # generate linearly separable dataset
            # this code is directly adapted from ARGD: https://github.com/aswilson07/ARGD
            np.random.seed(10)
            num_observations = 10240
            clusters = 1
            dim = 100
            spread = 20
            margin = np.random.randint(10, size=dim) / 10

            for i in range(clusters):
                if i == 0:
                    x1 = np.random.multivariate_normal(np.random.randint(spread, size=dim) + margin, np.identity(dim), num_observations)
                    x2 = np.random.multivariate_normal(np.random.randint(spread, size=dim), np.identity(dim), num_observations)
                else:
                    x1 = np.append(x1, np.random.multivariate_normal(np.random.randint(spread, size=dim) + margin, np.identity(dim), num_observations), axis=0)
                    x2 = np.append(x2, np.random.multivariate_normal(np.random.randint(spread, size=dim), np.identity(dim), num_observations), axis=0)
            X = np.zeros((2 * num_observations, dim))
            simulated_labels = np.zeros(X.shape[0])
            X[list(range(0, 2*num_observations, 2)),:] = x1
            X[list(range(1, 2*num_observations+1, 2)),:] = x2
            simulated_labels[list(range(0, 2*num_observations, 2))] = 1
            intercept = np.ones((X.shape[0], 1))
            simulated_separableish_features = np.hstack((intercept, X))
            dim += 1
            X = simulated_separableish_features
            y = simulated_labels
        elif name == 'protein':
            raw_data = load_svmlight_file(DATA_DIR + name + '.t')
            X, y = raw_data[0], raw_data[1]
            y -= 1
        else:
            raw_data = load_svmlight_file(DATA_DIR + name + '.t')
            X, y = raw_data[0], raw_data[1]


    if normalized:
        scaler.fit(X)
        X_sacled = scaler.transform(X)
    else:
        X_sacled = X

    if scipy.sparse.issparse(X_sacled):
        X = X_sacled.tocoo()
        X = torch.sparse_coo_tensor([X.row.tolist(), X.col.tolist()],
                                    X.data.tolist()).to(torch.float).to(device)
    else:
        X = torch.from_numpy(X_sacled).to(torch.float).to(device)
    y = torch.from_numpy(np.maximum(y, 0)).to(torch.float).to(device)

    if full_gradient:
        batch_size = X.shape[0]
    X = X.to_dense()
    dataLoader = data.TensorDataset(X, y)
    return data.DataLoader(dataLoader, batch_size=batch_size, shuffle=True), X.shape[1]