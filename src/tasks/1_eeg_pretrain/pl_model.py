import pytorch_lightning as pl

from torch.optim import AdamW, Adam
import torchmetrics,os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import io, torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from base.models import Autoencoder
from base.autoencoder import AutoencoderKL
from base.utils import instantiate_from_config

def load_model(config,test_loader):
    config = '/home/wht/multimodal_brain/src/tasks/base/configs/autoencoder_kl_32x32x4.yaml'
    config = OmegaConf.load(config)
    model = instantiate_from_config(config['model'])
    return model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,test_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject = batch
        loss, pred = self.eeg(eeg)
        return loss, pred
    
    def training_step(self, batch, batch_idx):
        loss, pred = self(batch)
        self.log('train_loss', loss, sync_dist=True, batch_size=pred.shape[0])
        return loss
    
    def plot_recon_eeg(self,data,save_path,pred_raw):
        fig, axes = plt.subplots(17, 1, figsize=(10, 34),dpi=300)
        # start_indices = np.where(np.diff(mask[0].astype(int)) == -1)[0] + 1
        # end_indices = np.where(np.diff(mask[0].astype(int)) == 1)[0] + 1
        # length = data.shape[1]
        # if not mask[0][0]:
        #     start_indices = np.insert(start_indices, 0, 0)
        # if not mask[0][-1]:
        #     end_indices = np.append(end_indices, length)

        for i in range(17):
            axes[i].plot(data[i], label='Raw Data',color='blue',linewidth=1)  
            axes[i].plot(pred_raw[i], label='Recon Data',color='orange',linewidth=1)
            # axes[i].set_title(f'Channel {i+1}')
            # axes[i].set_xlim([-1, 250]) 
            # axes[i].grid(True)
            # for start, end in zip(start_indices, end_indices):
            #     axes[i].axvline(start, color='green', linestyle='--')
            #     axes[i].axvline(end, color='green', linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, format='png')
        
    def validation_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject = batch
        loss, pred = self(batch)
        self.val_loss.update(loss)

        if batch_idx == 0:
            _target_raw = eeg[0].cpu().detach().numpy()
            image_path = os.path.join(self.logger.log_dir, f'Val_Recon_EEG_{self.current_epoch}.png')
            _pred_raw = self.eeg.unpatchify(pred)[0].cpu().detach().numpy()
            # _mask = mask[0].unsqueeze(1).expand(10, 425).reshape(10, 17,25).permute(1, 0, 2).reshape(17, -1).bool().cpu().numpy()
            self.plot_recon_eeg(_target_raw,image_path,_pred_raw)
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log('Val_loss', self.val_loss, sync_dist=True)
        self.val_loss.reset()
        
    def test_step(self,batch, batch_idx):
        loss, pred = self(batch)
        self.test_loss.update(loss)
        return loss
        
    def on_test_epoch_end(self):
        self.log('Test_loss', self.test_loss, sync_dist=True)
        self.test_loss.reset()
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'],weight_decay=self.config['train']['weight_decay'])
        return {'optimizer': optimizer}
  