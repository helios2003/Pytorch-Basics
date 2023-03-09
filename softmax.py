import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([1, 2, 3, 4])
output = softmax(x)
print(output)
x = torch.tensor([1, 2, 3, 4], dtype=torch.float)
output = torch.softmax(x, dim=0)
print(output)