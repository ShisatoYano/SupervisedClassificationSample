# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import axes3d

def logistic_regression_2d(x_0, x_1, w):
    y = 1 / (1 + np.exp(-(w[0] * x_0 + w[1] * x_1 + w[2])))
    return y

def show_3d_logistic_2d(ax, x_rng_0, x_rng_1, w):
    x_n = 50
    x_0 = np.linspace(x_rng_0[0], x_rng_0[1], x_n)
    x_1 = np.linspace(x_rng_1[0], x_rng_1[1], x_n)
    xx_0, xx_1 = np.meshgrid(x_0, x_1)
    y = logistic_regression_2d(xx_0, xx_1, w)
    ax.plot_surface(xx_0, xx_1, y, color='blue', edgecolor='gray',
                    rstride=5, cstride=5, alpha=0.3)

def show_3d_data(ax, x, t):
    c = [[0.5, 0.5, 0.5], [1, 1, 1]]
    for i in range(2):
        ax.plot(x[t[:,i]==1, 0], x[t[:,i]==1, 1], 1-i,
                marker='o', color=c[i], markeredgecolor='black',
                linestyle='none', markersize=5, alpha=0.8)
    ax.view_init(elev=25, azim=-30)

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

def show_contour_logistic_2d(x_rng_0, x_rng_1, w):
    x_n = 30
    x_0 = np.linspace(x_rng_0[0], x_rng_0[1], x_n)
    x_1 = np.linspace(x_rng_1[0], x_rng_1[1], x_n)
    xx_0, xx_1 = np.meshgrid(x_0, x_1)
    y = logistic_regression_2d(xx_0, xx_1, w)
    cont = plt.contour(xx_0, xx_1, y, levels=(0.2, 0.5, 0.8),
                       colors=['k', 'cornflowerblue', 'k'])
    cont.clabel(fmt='%1.1f', fontsize=10)
    plt.grid(True)

def cross_entropy_error_2d(w, x, t):
    x_n = x.shape[0]
    y = logistic_regression_2d(x[:, 0], x[:, 1], w)
    error = 0
    for n in range(len(y)):
        error = error - (t[n,0]*np.log(y[n]) + (1-t[n,0])*np.log(1-y[n]))
    error = error / x_n
    return error

def derivative_cross_entropy_error_2d(w, x, t):
    x_n = x.shape[0]
    y = logistic_regression_2d(x[:, 0], x[:, 1], w)
    error_array = np.zeros(3)
    for n in range(len(y)):
        error_array[0] = error_array[0] + ((y[n]-t[n, 0])*x[n, 0])
        error_array[1] = error_array[1] + ((y[n]-t[n, 0])*x[n, 1])
        error_array[2] = error_array[2] + (y[n]-t[n, 0])
    error_array = error_array / x_n
    return error_array

def fit_logistic_2d(w_init, x, t):
    result = minimize(cross_entropy_error_2d, w_init, args=(x, t),
                      jac=derivative_cross_entropy_error_2d,
                      method='CG')
    return result.x

if __name__ == "__main__":
    
    # load database
    data  = np.load('dim_2_class_2_data.npz')
    X     = data['X']
    T_2   = data['T']
    x_rng_0 = data['X_rng_0']
    x_rng_1 = data['X_rng_1']

    # searching parameter
    plt.figure(1, figsize=(7, 3))
    plt.subplots_adjust(wspace=0.5)

    Ax = plt.subplot(1, 2, 1, projection='3d')
    W_init = [-1, 0, 0]
    W = fit_logistic_2d(W_init, X, T_2)
    print("w0 = {0:.2f}, w1 = {1:.2f}, w2 = {2:.2f}".format(W[0], W[1], W[2]))
    show_3d_logistic_2d(Ax, x_rng_0, x_rng_1, W)
    show_3d_data(Ax, X, T_2)
    Ax.set_xlabel('$X_0$')
    Ax.set_ylabel('$X_1$')
    Ax.set_zlabel('Label')
    err = cross_entropy_error_2d(W, X, T_2)
    print("Cross entropy error = {0:.2f}".format(err))

    Ax = plt.subplot(1, 2, 2)
    show_2d_data(X, T_2)
    show_contour_logistic_2d(x_rng_0, x_rng_1, W)

    plt.show()