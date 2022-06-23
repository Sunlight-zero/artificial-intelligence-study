import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_data(x_dim, num_data, use_bias=True, sigma=1):
    # Generate weight and bias
    weight = np.random.normal(size=x_dim)
    weight = weight / np.linalg.norm(weight)
    if use_bias:
        bias = np.random.normal()
    else:
        bias = 0
    
    # Generate samples randomly.
    x = np.random.normal(loc=-bias * weight, scale=sigma, size=(num_data, x_dim))
    y = np.where(weight @ x.T + bias>=0, 1, -1)

    return x, y

if __name__ == '__main__':
    x, y = generate_data(x_dim=2, num_data=20)

    plt.scatter(x.T[0], x.T[1], c=y)
    plt.show()

    dataset = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    
    columns = ['x' + str(i + 1) for i in range(2)] + ['y']
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv('linear_separable_dataset.csv')
