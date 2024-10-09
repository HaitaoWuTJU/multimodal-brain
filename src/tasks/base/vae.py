import torch
import torch.nn as nn
from torch import nn, einsum
from einops import rearrange, repeat
import pytorch_lightning as pl
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

from .models import ProjectLayer
from .autoencoder import DiagonalGaussianDistribution
from .utils import instantiate_from_config

class VAE(pl.LightningModule):
    def __init__(self,to_mean=False, z_dim = 512,c_num = 17,lr=4.5e-6, kl_weight = 1e-2, timesteps=250,dummy_dim=3):
        super().__init__()
        self.encoder = ProjectLayer(c_num*timesteps,2*z_dim)
        self.decoder = ProjectLayer(z_dim,c_num*timesteps)

        self.automatic_optimization = False
        self.to_mean = to_mean
        self.z_dim = z_dim
        self.c_num = c_num
        self.lr = lr
        self.kl_weight = kl_weight
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def encode(self, x, scale):
        z = self.encoder(x)
        posterior = DiagonalGaussianDistribution(z, scale)
        return posterior

    def decode(self, z):
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True,scale=1.0):
        posterior = self.encode(input, scale)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon = self.decode(z[:,:self.z_dim])
        recon = recon.view(input.shape)
        return posterior, z, recon

    def compute_loss(self,target,recon,posterior):
        recon_loss = F.mse_loss(recon, target, reduction='sum')
        kl_loss = torch.sum(posterior.kl())/ target.shape[0]
        loss = recon_loss + self.kl_weight * kl_loss
        return loss
    
    def training_step(self, batch, batch_idx):

        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        inputs = eeg
        posterior, z, recon  = self(inputs)

        opt1 = self.optimizers()
        
        if self.to_mean:
            target = eeg_mean
        else:
            target = inputs
        
        aeloss = self.compute_loss(target,recon,posterior)
        self.manual_backward(aeloss)
        opt1.step()
        opt1.zero_grad()
        self.log("train_aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch[0].shape[0])

    def plot_recon_eeg(self,raw,raw_mean,raw_pred,save_path):
        fig, axes = plt.subplots(self.c_num, 1, figsize=(10, 34),dpi=300)
        for i in range(self.c_num):
            axes[i].plot(raw[i], label='Raw Data',color='blue',linewidth=1)  
            axes[i].plot(raw_mean[i], label='Raw Mean Data',color='red',linewidth=1)  
            axes[i].plot(raw_pred[i], label='Recon Data',color='orange',linewidth=1)
        plt.tight_layout()
        plt.savefig(save_path, format='png')

    def validation_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        inputs = eeg
        posterior, z, recon  = self(inputs)

        if self.to_mean:
            target = eeg_mean
        else:
            target = inputs
            
        aeloss = self.compute_loss(target,recon,posterior)
        self.log("val_aeloss", aeloss, prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch[0].shape[0])

        if batch_idx == 0:
            ele_inputs = eeg[0].cpu().detach().numpy()
            ele_mean = eeg_mean[0].cpu().detach().numpy()
            image_path = os.path.join(self.logger.log_dir, f'Val_Recon_EEG_{self.current_epoch}.png')
            ele_pred = recon[0].cpu().detach().numpy()
            self.plot_recon_eeg(ele_inputs, ele_mean, ele_pred,image_path)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.lr
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
       
        return [opt_ae]

    def get_last_layer(self):
        return self.decoder.conv_out.weight



