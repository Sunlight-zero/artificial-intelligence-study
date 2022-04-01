import numpy as np
import torch

a = np.array([[1, 1], [0, 1]])

x = torch.from_numpy(a)

print(x @ x)
