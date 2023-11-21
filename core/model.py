import torch
from torch import nn, functional

class CNN(nn.Module):
    def __init__(self, num_classes = 64):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, (3, 3), padding='same')
        self.relu = nn.ReLU()
        
        self.linear1 = nn.Linear(8*8*32, 128)
        
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = x.reshape(-1, 8 * 8 * 32)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x