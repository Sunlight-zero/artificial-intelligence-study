import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt


path = './didl/data'
batch_size = 256

# Define the LeNet.
leNet = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10)
)

# Import the dataset
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(
    path, train=True, transform=trans
)
mnist_test = torchvision.datasets.MNIST(
    path, train=False, transform=trans
)
train_iter = data.DataLoader(mnist_train, batch_size)
test_iter = data.DataLoader(mnist_test, batch_size)

# Define some useful function
def init_weights(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
        nn.init.xavier_uniform_(layer.weight)

def count_correct_classification(y_hat, y):
    """
    Count the correct answer between a one-hot code and a number.
    """
    y_hat = torch.argmax(y_hat, dim=1)
    cnt = (y_hat.type(y.dtype) == y)
    return float(torch.sum(cnt.type(y.dtype)))

def evaluate_accuracy(net: nn.Module, test_iter, device=None):
    net.eval()

    correct_cnt = 0
    total_cnt = 0

    for x, y in test_iter:
        x = x.to(device)
        y = y.to(device)
        y_hat = net(x)
        correct_cnt += count_correct_classification(y_hat, y)
        total_cnt += y.numel()
    
    return correct_cnt / total_cnt

# Start training
epochs= 30
lr = 0.5
leNet.apply(init_weights)
gpu = torch.device('cuda:0')
leNet.to(gpu)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(leNet.parameters(), lr)
for epoch in range(epochs):
    leNet.train()
    for x, y in train_iter:
        optimizer.zero_grad()
        x = x.to(gpu)
        y = y.to(gpu)
        y_hat = leNet(x)
        l = loss(y_hat, y)
        l.backward()
        optimizer.step()
    
    acc = evaluate_accuracy(leNet, test_iter, device=gpu)
    print(f"Epoch: {epoch + 1}, Test ACC: {acc}")
