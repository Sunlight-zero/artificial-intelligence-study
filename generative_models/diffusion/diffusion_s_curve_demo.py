"""
Use diffusion model to generate S-curve samples.
Just a demo.
Code is modified from this article: https://zhuanlan.zhihu.com/p/572770333
"""

from typing import Optional
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


DATASET_SIZE = 1000
NUM_STEP = 100


class SCurveDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = torch.Tensor(data)
        self.len = len(data)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

def get_s_curve_dataset(size: int=DATASET_SIZE, show: bool=False) -> torch.utils.data.Dataset:
    from sklearn.datasets import make_s_curve
    x, _ = make_s_curve(size, noise=0.1)
    data = x[:, [0, 2]]
    if show:
        plt.scatter(*data.T, s=5, marker='.')
        plt.show()
    dataset = SCurveDataset(data)
    return dataset

# Parameters of diffusion model
def init_parameters():
    global betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
    betas = torch.linspace(0.0001, 0.02, NUM_STEP)
    alphas = 1 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_bar.sqrt()
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_bar)

def get_x_t(x_0: torch.Tensor, t: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
    """
    Forward function.
    Get x_(t+1) from x_0, t and given noise e_t.
    """
    t = t.unsqueeze(-1)
    x_t = alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * e_t
    return x_t

class DiffusionNet(nn.Module):
    def __init__(self, input_dim: int, num_steps: int):
        """
        Network to predict noise
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.embedding1 = nn.Embedding(num_steps, 16)
        self.fc23 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        hidden = torch.cat([self.fc1(x), self.embedding1(t)], dim=1)
        return self.fc23(hidden)

def train(net: nn.Module, dataset: torch.utils.data.Dataset, batch_size: int,
          epochs: int, lr: float, device: torch.device,
          save_path: Optional[str]=None) -> None:
    loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    mse = nn.MSELoss()
    global betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
    net.to(device)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr)
    

    for epoch in range(epochs):
        sum_loss = 0.0
        cnt = 0

        for x_0 in loader:
            optimizer.zero_grad()
            x_0 = x_0.to(device)
            t = torch.randint(0, NUM_STEP, size=(batch_size, )).long().to(device)
            e_t = torch.randn_like(x_0).to(device)
            x_t = get_x_t(x_0, t, e_t)
            e_hat = net(x_t, t)
            loss = mse(e_hat, e_t)
            loss.backward()
            optimizer.step()

            sum_loss += float(loss)
            cnt += 1
        
        with torch.no_grad():
            print("Epoch {}, avg loss {}".format(epoch + 1, sum_loss / cnt))
    
    if save_path:
        torch.save(net, save_path)

def generate(net: nn.Module, num: int, device: torch.device) -> torch.Tensor:
    """
    Generate new samples
    """
    net.to(device)
    global betas, alphas, alphas_bar, alphas_bar_sqrt, one_minus_alphas_bar_sqrt
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)
    alphas_bar_sqrt = alphas_bar_sqrt.to(device)
    one_minus_alphas_bar_sqrt = one_minus_alphas_bar_sqrt.to(device)

    net.eval()
    with torch.no_grad():
        x_t = torch.randn(num, 2).to(device)
        for t in reversed(range(NUM_STEP)):
            t = t * torch.ones(num).long().to(device)
            e_hat = net(x_t, t)
            t.unsqueeze_(-1)
            mean = 1 / alphas[t].sqrt() * (x_t - betas[t] / one_minus_alphas_bar_sqrt[t] * e_hat)
            z_bar_t = torch.randn(num, 2).to(device)
            x_t = mean + betas[t] * z_bar_t
        return x_t.to(torch.device('cpu'))

if __name__ == "__main__":
    dataset = get_s_curve_dataset()
    gpu = torch.device('cuda')
    init_parameters()
    net = DiffusionNet(input_dim=2, num_steps=NUM_STEP)
    batch_size = 10
    lr = 1e-3
    epochs = 100
    train(net, dataset, batch_size, epochs, lr, device=gpu, save_path="./net.pkl")
    net = torch.load("./net.pkl")

    print("Start generating new samples")
    samples = generate(net, num=1000, device=gpu)
    plt.scatter(*samples.T, s=5, marker='.')
    plt.show()
