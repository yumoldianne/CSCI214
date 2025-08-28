# Pandas + OOP 1

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CSVLoader:
    def __init__(self, path):
        self.path = path
    def load(self):
        return pd.read_csv(self.path)

class DataCleaner:
    def __init__(self, fill_value=0):
        self.fill_value = fill_value
    def clean(self, df):
        # fill na and drop duplicates
        return df.fillna(self.fill_value).drop_duplicates()

class Aggregator:
    def __init__(self, group_col):
        self.group_col = group_col
    def aggregate(self, df):
        return df.groupby(self.group_col).mean(numeric_only=True)

# Example DataFrame created in-memory to demo
df_demo = pd.DataFrame({
    "category": ["A", "B", "A", None],
    "value": [10, 20, None, 5],
})
print("Original DF:\n", df_demo)
cleaner = DataCleaner(fill_value=0)
df_clean = cleaner.clean(df_demo)
print("Cleaned DF:\n", df_clean)
agg = Aggregator("category")
print("Aggregated:\n", agg.aggregate(df_clean))
print()