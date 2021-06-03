import numpy as np


def positive_mse_loss(beta, X, Y):
    p = np.poly1d(beta)
    error = sum([positive_mse(p(X[i]), Y[i]) for i in range(X.size)]) / X.size
    return (error)


def reg_positive_mse(y_pred, y_true):
    loss_pos = (y_pred - y_true) ** 2
    loss_neg = np.dot((y_pred - y_true < 0), (y_pred - y_true) ** 8)
    loss = loss_pos + loss_neg
    return np.sum(loss) / y_true.size


def reg_pos_loss(beta, X, Y, function):
    return positive_mse(function(X, beta), Y)


def mse_loss(beta, X, Y, function):
    return sum((function(X, beta) - Y) ** 2) / Y.size


def poly4(x_data, beta):
    p = np.poly1d(beta)
    return p(x_data)


def positive_mse(y_pred, y_true):
    loss = np.dot((10 ** 8 * (y_pred - y_true > 0) + np.ones(y_true.size)).T, (y_pred - y_true) ** 2)
    return np.sum(loss) / y_true.size


def pos_mse_loss(beta, X, Y, function):
    return positive_mse(function(X, beta), Y)


def reg_positive_mse(y_pred, y_true):
    loss_pos = (y_pred - y_true) ** 2
    loss_neg = np.dot((y_pred - y_true > 0), np.abs((y_pred - y_true)))
    loss = loss_pos + loss_neg
    return np.sum(loss) / y_true.size


def reg_pos_mse_loss(beta, X, Y, function):
    return reg_positive_mse(function(X, beta), Y)
