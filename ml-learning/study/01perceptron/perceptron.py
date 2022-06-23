import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir('./ml-learning/learning/perceptron/')

dataset = pd.read_csv('linear_separable_dataset.csv',
                      header=0, index_col=0).values
# x is the features and y is the classification (-1 or 1).
data_x, data_y = dataset[:, :-1], dataset[:, -1]
x_dim = data_x.shape[1]

weight = np.random.normal(size=x_dim)
bias = np.random.normal()

# Gradient descent:
lr = 0.1
epochs = 200
flag = False
is_success = False
for _ in range(epochs):
    flag = True
    for x, y in zip(data_x, data_y):
        if y * (weight @ x + bias) < 0:
            weight += lr * y * x
            bias += lr * y
            flag = False
    if flag:
        is_success = True
        break

print(weight, bias)

if x_dim == 2:
    xlim = [-2, 2]
    ylim = [- (weight[0] * x + bias) / weight[1] for x in xlim]
    plt.scatter(data_x.T[0], data_x.T[1], c=data_y)
    plt.plot(xlim, ylim)
    plt.show()
