# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def logistic_regression_3class(x_0, x_1, w):
    K = 3
    w = w.reshape((3, 3))
    n = len(x_1)
    y = np.zeros((n, K))
    for k in range(K):
        y[:, k] = np.exp(w[k,0]*x_0 + w[k,1]*x_1 + w[k,2])
    u = np.sum(y, axis=1)
    y = (y.T / u).T
    return y

def cross_entropy_error_3class(w, x, t):
    x_n = x.shape[0]
    y = logistic_regression_3class(x[:, 0], x[:, 1], w)
    error = 0
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            error = error - (t[n,k] * np.log(y[n,k]))
    error = error / x_n
    return error

def derivative_cross_entropy_error_3class(w, x, t):
    x_n = x.shape[0]
    y = logistic_regression_3class(x[:, 0], x[:, 1], w)
    error_array = np.zeros((3, 3))
    N, K = y.shape
    for n in range(N):
        for k in range(K):
            # x[n,:] includes x_0 and x_1
            # x_2 is always 1 and it is used as dummy input
            # these ones are integrated as 3 dimensional input
            error_array[k, :] = error_array[k, :] + (y[n,k] - t[n,k]) * np.r_[x[n,:], 1]
    error_array = error_array / x_n
    return error_array.reshape(-1)

def fit_logistic_3class(w_init, x, t):
    result = minimize(cross_entropy_error_3class, w_init, args=(x, t),
                      jac=derivative_cross_entropy_error_3class,
                      method='CG')
    return result.x

def show_contour_logistic_3class(x_rng_0, x_rng_1, w):
    x_n = 30
    x_0 = np.linspace(x_rng_0[0], x_rng_0[1], x_n)
    x_1 = np.linspace(x_rng_1[0], x_rng_1[1], x_n)
    xx_0, xx_1 = np.meshgrid(x_0, x_1)
    y = np.zeros((x_n, x_n, 3))
    for i in range(x_n):
        y_k = logistic_regression_3class(xx_0[:,i], xx_1[:,i], w)
        for j in range(3):
            y[:, i, j] = y_k[:, j]
    for j in range(3):
        cont = plt.contour(xx_0, xx_1, y[:, :, j],
                           levels=(0.5, 0.9),
                           colors=['cornflowerblue', 'k'])
        cont.clabel(fmt='%1.1f', fontsize=9)
    plt.grid(True)

def show_2d_data(x, t):
    w_k, K = t.shape
    color = [[0.5, 0.5, 0.5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k]==1, 0], x[t[:, k]==1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=color[k], alpha=0.8)
    plt.grid(True)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')

if __name__ == "__main__":
    
    # load database
    data  = np.load('dim_2_class_3_data.npz')
    X     = data['X']
    T_3   = data['T']
    x_rng_0 = data['X_rng_0']
    x_rng_1 = data['X_rng_1']

    # searching parameter
    W_init = np.zeros((3, 3))
    W = fit_logistic_3class(W_init, X, T_3)
    print(np.round(W.reshape((3,3)),2))
    err = cross_entropy_error_3class(W, X, T_3)
    print("Cross entropy error = {0:.2f}".format(err))

    plt.figure(figsize=(6, 6))
    show_2d_data(X, T_3)
    show_contour_logistic_3class(x_rng_0, x_rng_1, W)
    plt.show()