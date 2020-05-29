import itertools
from IPython.display import Image
from IPython import display
import matplotlib.pyplot as plt

import torch
from torch.utils import data

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

#trn_data = ??
#val_data = ??

#batch iterator
import cnn

batch_size = 64

#trn_dataloader = torch.utils.data.DataLoader(datasets=_dataset, batch_size=batch_size, shuffle=True)
#val_dataloader = torch.utils.data.DataLoader(datasets=_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)