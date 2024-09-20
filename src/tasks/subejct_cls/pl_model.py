import pytorch_lightning as pl

from torch.optim import AdamW, Adam
import torchmetrics,os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import io, torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from omegaconf import OmegaConf
import torch.nn as nn

from base.models import Autoencoder
from base._pipeline import StableDiffusionXLPipeline
from base.utils import instantiate_from_config

def load_model(config,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        if k == 'generation':
            model[k] = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        elif k == 'eeg':
            if v['target'] != 'base.autoencoder.AutoencoderKL':
                model[k] = instantiate_from_config(v)
            else:
                _config = '/home/wht/multimodal_brain/src/tasks/base/configs/autoencoder_kl_32x32x4.yaml'
                _config = OmegaConf.load(_config)
                model[k] = instantiate_from_config(_config['model'])
                ckpt = '/home/wht/multimodal_brain/src/tasks/1_eeg_pretrain/exp/VAE Pretrain/version_35/checkpoints/epoch=49-step=23650.ckpt'
                ckpt_pth = torch.load(ckpt)
                model[k].load_state_dict(ckpt_pth, strict=False)
        else:
            model[k] = instantiate_from_config(v)

    pl_model = PLModel(model,config,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,test_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        
        self.criterion = nn.CrossEntropyLoss()

        self.num_correct = 0
        self.cnt = 0
        
    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject,eeg_mean = batch
        
        scale = 1.0
        sample_posterior = False

        posterior = self.eeg.encode(eeg , scale)
        if sample_posterior:
            eeg_latent = posterior.sample()
        else:
            eeg_latent = posterior.mode()
        subject_logits = self.subject_cls(eeg_latent)

        loss = self.criterion(subject_logits, subject)
        return eeg_latent, subject_logits, loss
    
    def training_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, subject_logits, loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
     
    def validation_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, subject_logits, loss = self(batch)
        _, predicted = torch.max(subject_logits, 1)
        self.num_correct += (predicted == subject).sum().item()
        self.cnt += predicted.shape[0]
        return loss
    
    def on_validation_epoch_end(self):
        top_1_accuracy = self.num_correct/self.cnt
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
    
        self.num_correct = 0
        self.cnt = 0

    def configure_optimizers(self):
        # freeze eeg
        for param in self.eeg.parameters():
            param.requires_grad = False

        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'])
        return {'optimizer': optimizer}