import random
import numpy as np
import torch


np.random.seed(0)
N = 100  # number of points for each color
D = 2    # dimensionality of points
K = 3    # number of colors

X = np.zeros((N*K, D))
Y = np.zeros(N*K, dtype='uint8')

for i in range(K):
    j = list(range(N*i, N*(i+1)))
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(4*i, 4*(i+1), N) + np.random.randn(N)*0.2
    X[j] = np.c_[r*np.cos(t), r*np.sin(t)]
    Y[j] = i

np.save("data/train_samples", X)
np.save("data/train_labels", Y)

for i in range(K):
    j = list(range(N*i, N*(i+1)))
    r = np.linspace(0.0, 1.0, N)
    t = np.linspace(4*i, 4*(i+1), N) + np.random.randn(N)*0.2
    X[j] = np.c_[r*np.cos(t), r*np.sin(t)]

np.save("Data/test_samples", X)
