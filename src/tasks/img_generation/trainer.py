import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
import torch

class SimpleModel(pl.LightningModule):
    def __init__(self,config):
        super(SimpleModel, self).__init__()
        self.eeg_encoder = x

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = torch.relu(self.layer_1(x))
        x = torch.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)