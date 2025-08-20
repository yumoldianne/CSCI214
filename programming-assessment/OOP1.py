# OOP 1: Shape Hierarchy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class Shape:
    def area(self):
        raise NotImplementedError("Subclasses must implement area")

class Square(Shape):
    def __init__(self, side_length):
        # Create a tensor self.side with dtype float32
        self.side = torch.tensor(float(side_length), dtype=torch.float32)
    def area(self):
        return self.side ** 2

class Circle(Shape):
    def __init__(self, radius):
        self.radius = torch.tensor(float(radius), dtype=torch.float32)
    def area(self):
        return torch.pi * (self.radius ** 2)

sq = Square(4)
ci = Circle(3)
print("Square area:", sq.area().item())
print("Circle area:", ci.area().item())
print()