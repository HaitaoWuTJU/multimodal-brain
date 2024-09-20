import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class EEGNet(nn.Module):
    def __init__(self,num_ch, timesteps, output_dim):
        super(EEGNet, self).__init__()
        self.num_ch = num_ch
        self.timesteps = 250
        self.output_dim = output_dim
        
        # Layer 1
        self.conv1 = nn.Conv2d(1, 16, (1, num_ch), padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)
        
        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)
        
        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))
        
        self.fc1 = nn.Linear(1008, output_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        
        x = x.unsqueeze(1)
        
        x = x.permute(0, 1, 3, 2)

        # Layer 1
        x = F.elu(self.conv1(x))        
        x = self.batchnorm1(x)
        x = F.dropout(x, 0.25)
        
        x = x.permute(0, 3, 1, 2)
        
        # Layer 2
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, 0.25)
        x = self.pooling2(x)
        
        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, 0.25)
        # x = self.pooling3(x)
        # FC Layer
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x
    

class EEGFeatureExtractionNet(nn.Module):
    def __init__(self, input_channels, timesteps, output_dim=512):
        super(EEGFeatureExtractionNet, self).__init__()
        
        # 1. Spatial feature extraction using Conv1D
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # 2. Temporal feature extraction using LSTM
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=128, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 + 128 * timesteps, 512)  # Concatenating flattened Conv1D output and LSTM output
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        # Input shape: (batch_size, channels, timesteps)
        
        # 1. Spatial feature extraction with pooling
        spatial_features = torch.relu(self.conv1(x))  # (batch_size, 64, timesteps)
        spatial_features = self.pool1(spatial_features)  # (batch_size, 64, timesteps//2)
        spatial_features = torch.relu(self.conv2(spatial_features))  # (batch_size, 128, timesteps//2)
        spatial_features = self.pool2(spatial_features)  # (batch_size, 128, timesteps//4)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)  # Flatten for concatenation
        
        # 2. Temporal feature extraction
        x = x.permute(0, 2, 1)  # LSTM expects input shape: (batch_size, timesteps, channels)
        _, (temporal_features, _) = self.lstm(x)  # LSTM's output: (batch_size, 128)
        temporal_features = temporal_features.squeeze(0)  # Remove LSTM layer dimension
        
        # 3. Concatenate spatial and temporal features
        concatenated_features = torch.cat((spatial_features, temporal_features), dim=1)  # (batch_size, features)
        
        # 4. Fully connected layers
        x = torch.relu(self.fc1(concatenated_features))  # (batch_size, 512)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, output_dim)
        
        return x

# Example usage:
input_channels = 17   # Number of EEG channels
timesteps = 250       # Number of time steps
output_dim = 512      # Output dimension

# Create the model
model = EEGNet(input_channels, timesteps, output_dim)
# Example input
input_data = torch.randn(32, input_channels, timesteps)  # (batch_size, channels, timesteps)
output = model(input_data)
print(output.shape)  # Expected output shape: (32, 512)