# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    
    # load database
    data  = np.load('dim_1_class_2_data.npz')
    X_min = data['X_min']
    X_max = data['X_max']
    W = [8, -10]
    show_logistic_regression(X_min, X_max, W)
    plt.show()
