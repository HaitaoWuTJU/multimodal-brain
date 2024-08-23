import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import MultilabelClipLoss,SoftClipLoss


class PLModel(pl.LightningModule):
    def __init__(self, model,config):
        super().__init__()
        self.eeg_model = model['eeg']
        self.loss_fn = MultilabelClipLoss()

    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject = batch

        eeg_feature = self.eeg_model(eeg)
        logit_scale = self.eeg_model.logit_scale
        return eeg_feature, img_features, logit_scale
    
    def training_step(self, batch, batch_idx):
        eeg_feature, img_features, logit_scale = self(batch)
        loss = self.loss_fn(eeg_feature, img_features,logit_scale)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        eeg_feature, img_features, logit_scale = self(batch)
        loss = self.loss_fn(eeg_feature, img_features,logit_scale)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    
class StableDiffusionFineTuner(pl.LightningModule):
    def __init__(self, pipeline,eeg_model):
        super(StableDiffusionFineTuner, self).__init__()
        self.pipeline = pipeline
        self.eeg_model = eeg_model

    def forward(self, x):
        eeg_feature = self.eeg_model(x)
        print(eeg_feature.shape)
        output = self.pipeline(eeg_feature)
        return output

    def training_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject = batch
    
        outputs = self(eeg)
        loss = F.mse_loss(outputs, img)
        
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer