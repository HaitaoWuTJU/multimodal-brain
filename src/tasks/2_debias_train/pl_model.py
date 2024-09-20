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

from base.models import Autoencoder
from base._pipeline import StableDiffusionXLPipeline
from base.utils import instantiate_from_config
from lavis.models.clip_models.loss import ClipLoss

def load_model(config,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
        if k == 'generation':
            model[k] = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        elif k == 'eeg':
            if v['target'] != 'base.autoencoder.AutoencoderKL':
                model[k] = instantiate_from_config(v)
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain/version_19/checkpoints/epoch=99-step=16200.ckpt' #vae c17
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_63C/version_3/checkpoints/epoch=99-step=16200.ckpt' #vae c63
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_NoToMean/version_2/checkpoints/epoch=99-step=16200.ckpt'
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_ToMean/version_0/checkpoints/epoch=99-step=16200.ckpt'
                # ckpt_pth = torch.load(ckpt)
                # print(ckpt_pth)
                # model[k].load_state_dict(ckpt_pth['state_dict'], strict=False)
            else:
                _config = '/home/wht/multimodal_brain/src/tasks/base/configs/autoencoder_kl_32x32x4.yaml'
                _config = OmegaConf.load(_config)
                model[k] = instantiate_from_config(_config['model'])
                ckpt = '/home/wht/multimodal_brain/src/tasks/1_eeg_pretrain/exp/VAE Pretrain/version_35/checkpoints/epoch=49-step=23650.ckpt'
                ckpt_pth = torch.load(ckpt)
                model[k].load_state_dict(ckpt_pth['state_dict'], strict=False)
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
        
        self.criterion = ClipLoss()

        all_text_features=test_loader.dataset.all_text_features
        all_image_features=test_loader.dataset.all_image_features
        self.all_image_features = all_image_features/all_image_features.norm(dim=-1, keepdim=True)

        self.all_predicted_classes = []
        self.all_true_labels = []
        
    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject,eeg_mean = batch
        
        scale = 1.0
        sample_posterior = False

        posterior = self.eeg.encode(eeg , scale)
        if sample_posterior:
            eeg_latent = posterior.sample()
        else:
            eeg_latent = posterior.mode()
        debia_latent = self.debias(eeg_latent)

        logit_scale = self.debias.logit_scale
        loss = self.criterion(debia_latent, img_features, logit_scale)
        return eeg_latent, debia_latent, img_features, loss
    
    def training_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, debia_latent, img_features, loss = self(batch)

        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
     
    def validation_step(self, batch, batch_idx):
        self.all_image_features = self.all_image_features.to(self.device)
    
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, debia_latent, img_latent, loss = self(batch)

        debia_latent = debia_latent/debia_latent.norm(dim=-1, keepdim=True)

        similarity = (debia_latent @ self.all_image_features.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.all_true_labels.extend(label.cpu().numpy())
        return loss
    
    def on_validation_epoch_end(self):

        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self,batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, debia_latent, img_latent, loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])

        self.all_image_features = self.all_image_features.to(self.device)
        debia_latent = debia_latent/debia_latent.norm(dim=-1, keepdim=True)

        similarity = (debia_latent @ self.all_image_features.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        self.all_predicted_classes.append(top_k_indices.cpu().numpy())
        self.all_true_labels.extend(label.cpu().numpy())
        return loss
        
    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes,axis=0)
        all_true_labels = np.array(self.all_true_labels)
        
        top_1_predictions = all_predicted_classes[:, 0]
        top_1_correct = top_1_predictions == all_true_labels
        top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
        top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
        top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []
        
    def configure_optimizers(self):
        #freeze eeg
        # for param in self.eeg.parameters():
        #     param.requires_grad = False

        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'])
        return {'optimizer': optimizer}