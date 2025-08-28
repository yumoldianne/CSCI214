# Pandas + OOP 2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BaseColumnTransformer:
    def __init__(self, col):
        self.col = col
    def transform(self, df):
        raise NotImplementedError

class NormalizeColumn(BaseColumnTransformer):
    def transform(self, df):
        df = df.copy()
        col = self.col
        minv = df[col].min()
        maxv = df[col].max()
        denom = (maxv - minv)
        if denom == 0 or pd.isna(denom):
            df[col] = 0.0
        else:
            df[col] = (df[col] - minv) / denom
        return df

class EncodeCategory(BaseColumnTransformer):
    def transform(self, df):
        df = df.copy()
        df[self.col] = df[self.col].astype("category").cat.codes
        return df

# Demo
df_demo2 = pd.DataFrame({"val":[10,20,30], "cat":["A","B","A"]})
print("Before transform:\n", df_demo2)
norm = NormalizeColumn("val")
enc = EncodeCategory("cat")
df_demo2 = norm.transform(df_demo2)
df_demo2 = enc.transform(df_demo2)
print("After transforms:\n", df_demo2)
print()