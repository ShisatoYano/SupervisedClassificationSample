# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# data parameter
np.random.seed(seed=0)
X_min = 0
X_max = 2.5
X_n   = 50
X_color = ['cornflowerblue', 'gray']

# distribution parameter
dist_start = [0.4, 0.8]
dist_width = [0.8, 1.6]
Pi = 0.5

# input data
X = np.zeros(X_n)

# target label
T = np.zeros(X_n, dtype=np.uint8)

# generate data
for n in range(X_n):
    w_k = np.random.rand()
    T[n] = 0 * (w_k < Pi) + 1 * (w_k >= Pi)
    X[n] = np.random.rand() * dist_width[T[n]] + dist_start[T[n]]

# save as database
np.savez('dim_1_class_2_data.npz', X=X, X_min=X_min,
         X_max=X_max, X_n=X_n, X_color=X_color, T=T)

# print generated data
print('X = ' + str(np.round(X, 2)))
print('T = ' + str(T))

# plot generated data
plt.figure(figsize=(3, 3))
K = np.max(T) + 1
for k in range(K): # k = 0 or 1
    plt.plot(X[T==k], T[T==k], X_color[k], alpha=0.5,
             linestyle='none', marker='o')
plt.grid(True)
plt.ylim(-0.5, 1.5)
plt.xlim(X_min, X_max)
plt.xlabel('Weight X[g]')
plt.ylabel('Label T')
plt.yticks([0, 1])
plt.show()