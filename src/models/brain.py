import torch
import torch.nn as nn

class SharedEncoder(nn.Module):
    def __init__(self):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

class SemanticBranch(nn.Module):
    def __init__(self):
        super(SemanticBranch, self).__init__()
        self.fc1 = nn.Linear(in_features=64*input_length, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=semantic_dim)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EEGModel(nn.Module):
    def __init__(self):
        super(EEGModel, self).__init__()
        self.shared_encoder = SharedEncoder()
        self.semantic_branch = SemanticBranch()
        self.bias_branch = BiasBranch()
        
    def forward(self, x):
        shared_output = self.shared_encoder(x)
        semantic_output = self.semantic_branch(shared_output)
        bias_output = self.bias_branch(shared_output)
        return semantic_output, bias_output