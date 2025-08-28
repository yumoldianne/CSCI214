# OOP 2: Simple Neural Layers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BaseLayer(nn.Module):
    def forward(self, x):
        raise NotImplementedError

class DoubleLayer(BaseLayer):
    def forward(self, x):
        return x * 2

class AddBiasLayer(BaseLayer):
    def __init__(self, size):
        super().__init__()
        # create bias parameter of shape (size,)
        self.bias = nn.Parameter(torch.zeros(size))
    def forward(self, x):
        # x + bias (assumes bias shape is broadcastable to x)
        return x + self.bias

x = torch.tensor([1.0, 2.0, 3.0])
dl = DoubleLayer()
abl = AddBiasLayer(3)
print("Double:", dl(x))
print("AddBias (initial):", abl(x))
# update bias and show
with torch.no_grad():
    abl.bias += torch.tensor([0.1, 0.2, 0.3])
print("AddBias (after update):", abl(x))
print()
