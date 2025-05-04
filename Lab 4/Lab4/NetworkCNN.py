import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkCNN(nn.Module): 
    def __init__(self): 
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5) # input is color image with 3 channels
        # 32 is the number of feature maps and the kernel size is 5x5
 
        self.pool = nn.MaxPool2d(2,2)
        # maxpool will be used multiple times, but defined once
        # in_channels = 32 because self.conv1 output is 32 channels
        self.conv2 = nn.Conv2d(32,6,5) 
        # 53*53 comes from the dimension of the last conv layer
        self.fc1 = nn.Linear(6*53*53, 100) 
        self.fc2 = nn.Linear(100, 3)
        self.sm = nn.Softmax(dim=1)
 
    def forward(self, x): 
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6*53*53)
        x = F.relu(self.fc1(x))
        x = self.sm(self.fc2(x)) # softmax activation on final layer 
        return x
