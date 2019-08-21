# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
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

if __name__ == "__main__":
    Ax = plt.subplot(1, 1, 1, projection='3d')
    # load database
    data  = np.load('dim_2_class_2_data.npz')
    X   = data['X']
    T_2 = data['T']
    x_rng_0 = data['X_rng_0']
    x_rng_1 = data['X_rng_1']
    W = [-1, -1, -1]
    show_3d_logistic_2d(Ax, x_rng_0, x_rng_1, W)
    show_3d_data(Ax, X, T_2)
    Ax.set_xlabel('$X_0$')
    Ax.set_ylabel('$X_1$')
    Ax.set_zlabel('Label')
    plt.show()
