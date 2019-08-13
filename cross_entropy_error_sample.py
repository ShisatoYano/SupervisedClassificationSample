# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def logistic_regression(x, w):
    y = 1 / (1 + np.exp(-(w[0] * x + w[1])))
    return y

def cross_entropy_error(w, x, x_n, t):
    y = logistic_regression(x, w)
    error = 0
    for n in range(len(y)):
        error = error - (t[n]*np.log(y[n]) + (1-t[n])*np.log(1-y[n]))
    error = error / x_n
    return error

if __name__ == "__main__":
    
    # load data
    data  = np.load('dim_1_class_2_data.npz')
    X   = data['X']
    X_n = data['X_n']
    T   = data['T']

    # calculating error
    res_n = 80
    w_range = np.array([[0, 15], [-15, 0]])
    x_0 = np.linspace(w_range[0, 0], w_range[0, 1], res_n)
    x_1 = np.linspace(w_range[1, 0], w_range[1, 1], res_n)
    xx_0, xx_1 = np.meshgrid(x_0, x_1)
    error_array = np.zeros((len(x_1), len(x_0)))
    w = np.zeros(2)
    for i_0 in range(res_n):
        for i_1 in range(res_n):
            w[0] = x_0[i_0]
            w[1] = x_1[i_1]
            error_array[i_1, i_0] = cross_entropy_error(w, X, X_n, T)
    
    # show result
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(wspace=0.5)
    
    ax = plt.subplot(1, 2, 1, projection='3d')
    ax.plot_surface(xx_0, xx_1, error_array, color='blue', edgecolor='black',
                    rstride=10, cstride=10, alpha=0.3)
    ax.set_xlabel('$w_0$', fontsize=14)
    ax.set_ylabel('$w_1$', fontsize=14)
    ax.set_xlim(0, 15)
    ax.set_ylim(-15, 0)
    ax.set_zlim(0, 8)
    ax.view_init(30, -95)

    plt.subplot(1, 2, 2)
    contour = plt.contour(xx_0, xx_1, error_array, 20, colors='black',
                          levels=[0.26, 0.4, 0.8, 1.6, 3.2, 6.4])
    contour.clabel(fmt='%1.1f', fontsize=8)
    plt.xlabel('$w_0$', fontsize=14)
    plt.ylabel('$w_1$', fontsize=14)

    plt.grid(True)
    plt.show()