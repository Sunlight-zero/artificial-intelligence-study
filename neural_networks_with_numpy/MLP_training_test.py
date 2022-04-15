from mnist_loader import load_data_wrapper
from MLP import MLP


tr_d, va_d, te_d = load_data_wrapper()
net = MLP([28 * 28, 30, 10])
net.train(tr_d, 30, 10, 0.1, te_d)
