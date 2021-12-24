import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset 
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import psutil
from pathlib import Path
import datetime

device = 'cpu'

class Lin(nn.Module):
    def __init__(self):
        """ Define and instantiate your layers"""
        super(Lin, self).__init__()

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        res = F.relu(self.fc1(x))
        res = self.fc2(res)
        return res

net = Lin().to(device)
data = torch.tensor([1.0, 2.0], device=device)
print(net(data))
