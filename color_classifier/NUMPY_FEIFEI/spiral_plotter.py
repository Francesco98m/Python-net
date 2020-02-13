import numpy as np
import matplotlib.pyplot as plt


Xtr = np.load("data/train_samples" + '.npy')
Ytr = np.load("data/train_labels" + '.npy')
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr, s=40, cmap=plt.cm.Spectral)
plt.title("Training set")
plt.show()


Xtr = np.load("data/test_samples" + '.npy')
Ytr = np.load("data/test_scores" + '.npy')
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=Ytr, s=40, cmap=plt.cm.Spectral)
plt.title("Test")
plt.show()


