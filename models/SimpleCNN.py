# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import BaseModel

class SimpleCNN(BaseModel):
    def __init__(self, lr=1e-3, threshold=0.5):
        super().__init__(lr=lr, threshold=threshold)
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 7 * 7, 32)  # Adjusted to match output of pooling layer
        self.fc2 = nn.Linear(32, 1) # Output layer with 1 unit for binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 7 * 7) # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # Sigmoid activation for binary classification
        return x
