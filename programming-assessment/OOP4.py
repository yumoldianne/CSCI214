# OOP 4: Metric Calculation Framework

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class BaseMetric:
    def reset(self):
        raise NotImplementedError
    def update(self, preds, targets):
        raise NotImplementedError
    def compute(self):
        raise NotImplementedError

class AccuracyMetric(BaseMetric):
    def reset(self):
        self.correct = 0
        self.total = 0
    def update(self, preds, targets):
        preds = preds.argmax(dim=1)
        self.correct += (preds == targets).sum().item()
        self.total += targets.numel()
    def compute(self):
        if self.total <= 0:
            return 0.0
        return self.correct / self.total

class MAEMetric(BaseMetric):
    def reset(self):
        self.error_sum = 0.0
        self.total = 0
    def update(self, preds, targets):
        # preds and targets are tensors (preds may be raw floats)
        # accumulate sum of absolute errors
        self.error_sum += torch.abs(preds - targets).sum().item()
        self.total += targets.numel()
    def compute(self):
        if self.total <= 0:
            return 0.0
        return self.error_sum / self.total

acc = AccuracyMetric()
mae = MAEMetric()
preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])  # logits/probs over 2 classes
targets_cls = torch.tensor([1, 0])
targets_reg = torch.tensor([0.0, 1.0])

acc.reset(); acc.update(preds, targets_cls)
mae.reset(); mae.update(preds[:,0], targets_reg)

print("Accuracy:", acc.compute())
print("MAE:", mae.compute())
print()