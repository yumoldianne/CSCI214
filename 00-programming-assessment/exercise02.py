# Exercise 2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

labels = torch.tensor([3, 0, 1, 2, 1], dtype=torch.long)  # 5 labels in {0,1,2,3}
num_classes = 4
# One-hot
one_hot = F.one_hot(labels, num_classes=num_classes).to(torch.float32)
print("One-hot:\n", one_hot)
# Convert one-hot into probabilities using softmax along classes
probs = torch.softmax(one_hot, dim=1)
print("Softmax probs:\n", probs)
# Recover original labels (argmax)
recovered = probs.argmax(dim=1)
print("Recovered labels:", recovered.tolist())
assert torch.equal(recovered, labels)
print("Recovered equals original:", torch.equal(recovered, labels))
print()