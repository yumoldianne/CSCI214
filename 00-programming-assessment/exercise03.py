# Exercise 3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Create float64 random tensor on CPU
t = torch.randn((10, 10), dtype=torch.float64, device="cpu")
print("Original:", t.dtype, t.device)
# Move to GPU if available and convert to float16
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
t_dev = t.to(device=device, dtype=torch.float16)
print("Moved to device:", t_dev.device, "dtype:", t_dev.dtype)
# Multiply by 2.5 while on device
t_dev = t_dev * 2.5
# Bring back to CPU as float32
t_back = t_dev.to("cpu", dtype=torch.float32)
print("Back on CPU:", t_back.device, t_back.dtype)
print()
