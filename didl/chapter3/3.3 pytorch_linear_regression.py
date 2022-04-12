import torch
from torch import nn
from torch.utils import data
import numpy as np


def generate_data(w, b, n: int, noise_sigma=0.01):
    w = np.array(w)
    b = np.array(b)
    x = np.random.normal(0, 1, (n, len(w)))
    y = x @ w + b + np.random.normal(0, noise_sigma, b.shape)
    return torch.from_numpy(x),\
        torch.from_numpy(y.reshape((-1, 1)))

class LinearRegression:

    def __init__(self, input_dim, output_dim):
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim))
        self.net[0].weight.data.normal_(0, 0.01)
        self.net[0].bias.data.fill_(0)
        # Mean Square Error
        self.loss = nn.MSELoss()
        self.optimizer = None
    
    def load_array(self, data_tensors, batch_size, is_train=True):
        """
        Use the API in PyTorch to read the data
        """
        dataset = data.TensorDataset(*data_tensors)
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
    
    def train(self, dataset, mini_batch_size, eta: float, epochs: int):
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=eta)
        data_iter = self.load_array(dataset, mini_batch_size)
        for epoch in range(epochs):
            for x, y in data_iter:
                l = self.loss(self.net(x), y)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()
            l = self.loss(self.net(dataset[0]), dataset[1])
            print(f'epoch {epoch + 1} / {epochs}, loss {l:f}')
    
    @property
    def parameters(self):
        w = self.net[0].weight.data
        b = self.net[0].bias.data
        return w, b

if __name__ == '__main__':
    # 统一神经网络的数据格式，避免出现TypeError
    torch.set_default_tensor_type(torch.DoubleTensor)
    true_w, true_b = [2, 1.5], 1.0
    features, labels = generate_data(true_w, true_b, 1000, 0.01)
    lr = LinearRegression(2, 1)
    lr.train((features, labels), 10, 0.1, 5)
    w_hat, b_hat = lr.parameters
    print('Estimate:', w_hat, b_hat)
    print('True values:', true_w, true_b)

