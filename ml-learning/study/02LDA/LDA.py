import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('./ml-learning/study/02LDA/')

dataset = pd.read_csv('LDA_data.csv', index_col=0).values

x_dim = 2
# Separate two categories.
category_1 = dataset[dataset[:, -1] == 1][:, :-1]
category_2 = dataset[dataset[:, -1] == -1][:, :-1]

# Calculate the mean and the covariance matrix
mu_1, mu_2 = np.mean(category_1, axis=0), np.mean(category_2, axis=0)
S_1, S_2 = np.cov(category_1.T), np.cov(category_2.T)

# Calculate the project vector
u, s, v = np.linalg.svd(S_1 + S_2)
s_inv = v.T @ np.diag(1 / s) @ u.T
w = np.reshape(s_inv @ (mu_1 - mu_2), (-1, 1))

# Dimensionality reduction
new_category_1 = category_1 @ w
new_category_2 = category_2 @ w

plt.scatter(new_category_1, np.zeros_like(new_category_1), c='r')
plt.scatter(new_category_2, np.zeros_like(new_category_2), c='b')
plt.show()
