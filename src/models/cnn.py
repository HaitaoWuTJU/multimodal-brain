import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self,num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(64 * 8 * 8, 2048)

        # self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x): 
        
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)

        batch = x.shape[0]
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(batch, 64 * 8 * 8)
        x = self.fc1(x)
        # x = self.fc2(x)
        
        return x
    