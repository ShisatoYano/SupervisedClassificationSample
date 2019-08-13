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

