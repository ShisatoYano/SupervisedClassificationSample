# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def show_sample_data(x, t):
    w_k, K = t.shape
    color = [[0.5, 0.5, 0.5], [1, 1, 1], [0, 0, 0]]
    for k in range(K):
        plt.plot(x[t[:, k]==1, 0], x[t[:, k]==1, 1],
                 linestyle='none', markeredgecolor='black',
                 marker='o', color=color[k], alpha=0.8)
    plt.grid(True)
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')

# data parameter
np.random.seed(seed=1)
N = 200
K = 3
label_3d = np.zeros((N, 3), dtype=np.uint8)
label_2d = np.zeros((N, 2), dtype=np.uint8)
X         = np.zeros((N, 2))
X_range_0 = [-3, 3]
X_range_1 = [-3, 3]
Mu    = np.array([[-0.5, -0.5], [0.5, 1.0], [1, -0.5]])
Sigma = np.array([[0.8, 0.8], [0.7, 0.3], [0.3, 0.7]])
Pi    = np.array([0.4, 0.8, 1.0])

# generate data
for n in range(N):
    w_k = np.random.rand()
    for k in range(K):
        if w_k < Pi[k]:
            label_3d[n, k] = 1
            break
    for k in range(2):
        X[n, k] = (np.random.randn()*Sigma[label_3d[n,:]==1, k]+Mu[label_3d[n,:]==1, k])

label_2d[:, 0] = label_3d[:, 0]
label_2d[:, 1] = label_3d[:, 1] | label_3d[:, 2]

print(X[:5, :])
print(label_2d[:5, :])
print(label_3d[:5, :])

# save as database
np.savez('dim_2_class_2_data.npz', X=X, T=label_2d, N=N, K=K,
         X_rng_0=X_range_0, X_rng_1=X_range_1, Mu=Mu, Sigma=Sigma, Pi=Pi)
np.savez('dim_2_class_3_data.npz', X=X, T=label_3d, N=N, K=K,
         X_rng_0=X_range_0, X_rng_1=X_range_1, Mu=Mu, Sigma=Sigma, Pi=Pi)

# show data
plt.figure(figsize=(7.5, 5))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
show_sample_data(X, label_2d)
plt.xlim(X_range_0)
plt.ylim(X_range_1)
plt.subplot(1, 2, 2)
show_sample_data(X, label_3d)
plt.xlim(X_range_0)
plt.ylim(X_range_1)
plt.show()