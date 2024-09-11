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

from base.models import Autoencoder
from base._pipeline import StableDiffusionXLPipeline
from base.utils import MultilabelClipLoss

def load_model(config,test_loader):
    model = {}
    for k,v in config['models'].items():
        if k == 'generation':
            model[k] = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

        elif k == 'eeg':
            model[k] = globals()[v['name']](**v['args'])
            ckpt = '/home/wht/multimodal_brain/src/tasks/1_eeg_pretrain/exp/pretrain/version_11/checkpoints/epoch=49-step=16200.ckpt'
            ckpt_pth = torch.load(ckpt)
            model[k].load_state_dict(ckpt_pth, strict=False)
  
    pl_model = PLModel(model,config,test_loader)
    return pl_model

class PLModel(pl.LightningModule):
    def __init__(self, model,config,test_loader):
        super().__init__()

        self.config = config
        for key, value in model.items():
            setattr(self, f"{key}", value)
        
        self.criterion = MultilabelClipLoss()
        # self.image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(
        #     self.device, dtype=torch.float16
        #     )
        # self.clip_image_processor = CLIPImageProcessor()

        self.val_loss = torchmetrics.MeanMetric()
        self.test_loss = torchmetrics.MeanMetric()
        
    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject = batch
        
        # dtype = next(self.generation.image_encoder.parameters()).dtype
        # if not isinstance(img, torch.Tensor):
        #     img = self.generation.feature_extractor(img, return_tensors="pt").pixel_values

        # img = img.to(device=self.device, dtype=dtype)
        
        eeg_latent = self.eeg.forward_cls(eeg)
        img_latent = img_features
        print(eeg_latent.shape,img_latent.shape)
        
        debia_latent = self.debias(eeg_latent)
        
        return eeg_latent, debia_latent, img_latent
    
    def training_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject = batch
        eeg_latent, debia_latent, img_latent = self(batch)
    
        loss = self.criterion(debia_latent,img_latent)
        
        self.log('train_loss', loss, sync_dist=True, batch_size=eeg.shape[0])
        return loss
     
    def validation_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject = batch
        eeg_latent, debia_latent, img_latent, gen_img_latent, gen_img = self(batch)
        reconstruction_loss = torch.nn.functional.mse_loss(gen_img, img)
        latent_loss = torch.nn.functional.cosine_embedding_loss(img_latent, gen_img_latent, torch.ones(eeg_latent.size(0)).to(eeg_latent.device))
        loss = reconstruction_loss + latent_loss

        self.val_loss.update(loss)
        return loss
    
    def on_validation_epoch_end(self):
        self.log('Val_loss', self.val_loss, sync_dist=True)
        self.val_loss.reset()
        
    def test_step(self,batch, batch_idx):
        self.validation_step(batch,batch_idx)
        loss = self.test_loss.update(loss)
        return loss
        
    def on_test_epoch_end(self):
        self.log('Test_loss', self.test_loss, sync_dist=True)
        self.test_loss.reset()
        
    def configure_optimizers(self):

        #freeze eeg
        for param in self.eeg.parameters():
            param.requires_grad = False
        
        #freeze stable diffusion VAE
        for param in self.generation.vae.parameters():
            param.requires_grad = False

        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'], weight_decay = self.config['train']['weight_decay'])
        return {'optimizer': optimizer}