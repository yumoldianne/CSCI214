# OOP 3: Dataset Variants

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BaseDataset(Dataset):
    def __init__(self, data, labels):
        # store tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels)  # dtype left to default (int64)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class SquaredDataset(BaseDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return x ** 2, y

class NormalizedDataset(BaseDataset):
    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        # Normalize sample-wise
        # To avoid zero-division: if std==0, return zeros
        std = x.std()
        if std == 0:
            norm = x - x.mean()
        else:
            norm = (x - x.mean()) / std
        return norm, y

data = [[1,2],[3,4],[5,6]]
labels = [0,1,0]
sq_ds = SquaredDataset(data, labels)
norm_ds = NormalizedDataset(data, labels)
print("Squared first item:", sq_ds[0])
print("Normalized first item:", norm_ds[0])
print()