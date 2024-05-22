import torch
import torch.nn as nn
import torchvision.models as models


class Resnet18(nn.Module):
    def __init__(self, num_classes=1000, pretrained=True):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)

        if num_classes != 1000:
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)