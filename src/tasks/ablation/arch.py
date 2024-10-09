import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from diffusers import DiffusionPipeline
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import sys
import shutil
from pytorch_lightning.accelerators import find_usable_cuda_devices
import json
import numpy as np
import pytorch_lightning as pl
from torch.optim import AdamW, Adam
from lavis.models.clip_models.loss import ClipLoss

##import user lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.data import load_data
from base.utils import update_config
from base.utils import instantiate_from_config
#user env
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def load_model(config,test_loader):
    model = {}
    for k,v in config['models'].items():
        print(f"init {k}")
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
        
        self.kl_weight  = 0

    def forward(self, batch):
        eeg, label, img, img_features, text, text_features, session, subject,eeg_mean = batch
        
    
        eeg_latent = self.eeg.forward_cls(eeg)

        debia_latent = self.debias(eeg_latent[:,:512])

        logit_scale = self.debias.logit_scale
        clip_loss = self.criterion(debia_latent, img_features, logit_scale)
        

        loss = clip_loss
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
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
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

        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return  {'test_loss': avg_test_loss.item(), 'test_top1_acc': top_1_accuracy.item(),'test_top5_acc':top_k_accuracy.item()}
        
    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'])
        return {'optimizer': optimizer}
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/wht/multimodal_brain/src/tasks/base/configs/arch.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-08',
        help="the subjects",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]
    print(config)

    os.makedirs(config['save_dir'],exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name="Arch", version=f"{config['data']['subjects']}")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, test_loader = load_data(config)
    
    pl_model = load_model(config,test_loader)

    checkpoint_callback = ModelCheckpoint(
            monitor='val_top1_acc',
            mode='max',
            save_top_k=3)

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'], devices=[6],accelerator='cuda',logger=logger)
    print(trainer.logger.log_dir, trainer.logger.version)
    ckpt_path = "/home/wht/multimodal_brain/src/tasks/exp/VAE_finetune/version_4/checkpoints/epoch=99-step=1700.ckpt"
    ckpt_path = None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader,ckpt_path=ckpt_path)
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
     
if __name__=="__main__":
    main()