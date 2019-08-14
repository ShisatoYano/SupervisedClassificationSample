# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def show_sample_data(x, t, x_min, x_max, x_col):
    K = np.max(t) + 1
    for k in range(K): # k = 0 or 1
        plt.plot(x[t==k], t[t==k], x_col[k], alpha=0.5,
                linestyle='none', marker='o')
    plt.grid(True)
    plt.ylim(-0.5, 1.5)
    plt.xlim(x_min, x_max)
    plt.xlabel('Weight X[g]')
    plt.ylabel('Label T')
    plt.yticks([0, 1])

def logistic_regression(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

def show_logistic_regression(x_min, x_max, w):
    x_b = np.linspace(x_min, x_max, 100)
    y = logistic_regression(x_b, w)
    plt.plot(x_b, y, color='gray', linewidth=4)
    # decision boundary
    i = np.min(np.where(y > 0.5))
    B = (x_b[i-1] + x_b[i]) / 2
    plt.plot([B, B], [-0.5, 1.5], color='k', linestyle='--')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    return B

def cross_entropy_error(w, x, t, x_n):
    y = logistic_regression(x, w)
    error = 0
    for n in range(len(y)):
        error = error - (t[n]*np.log(y[n]) + (1-t[n])*np.log(1-y[n]))
    error = error / x_n
    return error

def derivative_cross_entropy_error(w, x, t, x_n):
    y = logistic_regression(x, w)
    error_array = np.zeros(2)
    for n in range(len(y)):
        error_array[0] = error_array[0] + ((y[n]-t[n])*x[n])
        error_array[1] = error_array[1] + (y[n]-t[n])
    error_array = error_array / x_n
    return error_array

def fit_logistic(w_init, x, t, x_n):
    result = minimize(cross_entropy_error, w_init, args=(x, t, x_n),
                      jac=derivative_cross_entropy_error,
                      method='CG')
    return result.x

if __name__ == "__main__":
    
    # load database
    data  = np.load('dim_1_class_2_data.npz')
    X     = data['X']
    T     = data['T']
    X_min = data['X_min']
    X_max = data['X_max']
    X_n   = data['X_n']
    X_col = data['X_color']

    # searching parameter
    plt.figure(1, figsize=(6, 6))
    W_init = [1, -1]
    W = fit_logistic(W_init, X, T, X_n)
    print("w0 = {0:.2f}, w1 = {1:.2f}".format(W[0], W[1]))
    B = show_logistic_regression(X_min, X_max, W)
    show_sample_data(X, T, X_min, X_max, X_col)
    plt.xlim(X_min, X_max)
    plt.ylim(-0.5, 1.5)
    err = cross_entropy_error(W, X, T, X_n)
    print("Cross entropy error = {0:.2f}".format(err))
    print("Boundary = {0:.2f} g".format(B))
    plt.show()