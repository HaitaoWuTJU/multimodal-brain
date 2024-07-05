import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGEncoder(nn.Module):
    def __init__(self):
        super(EEGEncoder, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv1d(in_channels=63, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1)

        # Define max pooling layer
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        
        self.avg_pool =  nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(512,512)
        # self.fc2 = nn.Linear(1024, 512)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        batch = x.shape[0]
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = self.avg_pool(x)
        x = x.view(batch,-1)
        x = self.fc1(x)
        x = x.view(batch,-1)
        return x
    
class LSTMModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_layers=2, feature_size=512):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, feature_size)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
if __name__=="__main__":
    # model = EEGEncoder()
    model = LSTMModel()
    input_data = torch.randn(2, 63, 250)  # Example input data, batch size 1
    output_feature_vector = model(input_data)
    print(output_feature_vector.shape)  # Output shape: (1, 512