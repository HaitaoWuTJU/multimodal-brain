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
from lavis.models.clip_models.loss import ClipLoss
import open_clip

from base.utils import instantiate_from_config, SoftClipLoss

def load_model(config,test_loader):
    model = {}
    for k,v in config['models'].items():
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

        model_type = 'ViT-B-32'
        pretrain_map={
                        'RN50':'openai',
                        'RN101':'openai',
                        'RN50x4':'openai',
                        'ViT-B-16':'laion2b_s34b_b88k',
                        'ViT-B-32':'laion2b_s34b_b79k',
                        'ViT-L-14':'laion2b_s32b_b82k',
                        'ViT-H-14':'laion2b_s32b_b79k',
                        'ViT-g-14':'laion2b_s34b_b88k', 
                        'ViT-bigG-14':'laion2b_s39b_b160k'}
        self.vlmodel = {}
        self.vlmodel['model'] = open_clip.create_model_and_transforms(model_type, device="cuda:7",pretrained=pretrain_map[model_type])[0]
        self.vlmodel['model'].eval()
        self.vlmodel['model'].visual.output_tokens = True
        self.vlmodel['model'].visual.attn_pool = self.vlmodel['model'].visual.proj = None
        
    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        
        pooler_output, last_hidden_state = self.vlmodel['model'].visual(img)
        last_hidden_state = torch.cat([pooler_output.unsqueeze(1), last_hidden_state], dim=1).view(eeg.shape[0],-1)

        semantic_features = self.projcect(eeg)
        logit_scale = self.projcect.logit_scale #.exp()
        loss = self.criterion(semantic_features, last_hidden_state, logit_scale)

        return loss, semantic_features, last_hidden_state
    
    def training_step(self, batch, batch_idx):
        loss, semantic_features, last_hidden_state = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        loss, semantic_features, last_hidden_state = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
    
        semantic_features = semantic_features/semantic_features.norm(dim=-1, keepdim=True)
        last_hidden_state = last_hidden_state/last_hidden_state.norm(dim=-1, keepdim=True)

        similarity = (semantic_features @ last_hidden_state.T)
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
        # print(f"Epoch:{self.current_epoch}, top1-acc: {top_1_accuracy:.3f}, top5-acc: {top_k_accuracy:.3f}")
        self.all_predicted_classes = []
        self.all_true_labels = []
        
    def test_step(self,batch, batch_idx):
        eeg, label, img, img_features, text, text_features, session, subject, eeg_mean = batch
        loss, semantic_features, last_hidden_state = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])


        self.all_image_features = self.all_image_features.to(self.device)
        semantic_features = semantic_features/semantic_features.norm(dim=-1, keepdim=True)
        last_hidden_state = last_hidden_state/last_hidden_state.norm(dim=-1, keepdim=True)

        similarity = (semantic_features @ last_hidden_state.T)
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
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr=self.config['train']['lr'])#, weight_decay=self.config['train']['weight_decay'])
        return {'optimizer': optimizer}