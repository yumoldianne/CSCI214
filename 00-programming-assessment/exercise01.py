# Exercise 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Create an int32 tensor of shape (2,3,4) with values in [-100,100]
t_int = torch.randint(-100, 101, (2, 3, 4), dtype=torch.int32)
print("Original:", t_int.shape, t_int.dtype)
# Convert to float32
t_f32 = t_int.to(torch.float32)
# Scale to [0,1] given original range [-100, 100]: (x + 100) / 200
t_scaled = (t_f32 + 100.0) / 200.0
# Flatten into 1D
t_flat = t_scaled.view(-1)
print("After convert -> dtype:", t_scaled.dtype, "min/max:", t_scaled.min().item(), t_scaled.max().item())
print("Flattened shape:", t_flat.shape)
print()
