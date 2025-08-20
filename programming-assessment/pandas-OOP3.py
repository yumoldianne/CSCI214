# Pandas + OOP 3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class DataFrameManager:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

class Merger:
    def __init__(self, key):
        self.key = key
    def merge(self, manager):
        # Use pandas.merge
        return pd.merge(manager.df1, manager.df2, on=self.key)

class Filter:
    def __init__(self, col, threshold):
        self.col = col
        self.threshold = threshold
    def filter(self, df):
        return df[df[self.col] > self.threshold]

# Demo
df1 = pd.DataFrame({"id":[1,2,3],"val1":[10,20,30]})
df2 = pd.DataFrame({"id":[1,2,3],"val2":[5,25,35]})
mgr = DataFrameManager(df1, df2)
merger = Merger("id")
flt = Filter("val2", 20)
merged = merger.merge(mgr)
print("Merged:\n", merged)
print("Filtered:\n", flt.filter(merged))