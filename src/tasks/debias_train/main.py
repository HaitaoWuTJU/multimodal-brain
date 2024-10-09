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
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import open_clip
# from lavis.models.clip_models.loss import ClipLoss

##import user lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.data import load_data
from base.utils import update_config, SoftClipLoss , ClipLoss
from base._pipeline import StableDiffusionXLPipeline
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
        if k == 'generation':
            model[k] = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
        elif k == 'eeg':
            if v['target'] != 'base.autoencoder.AutoencoderKL':
                model[k] = instantiate_from_config(v)
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain/version_19/checkpoints/epoch=99-step=16200.ckpt' #vae c17
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_63C/version_3/checkpoints/epoch=99-step=16200.ckpt' #vae c63
                # ckpt = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_NoToMean/version_2/checkpoints/epoch=99-step=16200.ckpt'
                # ckpt_path = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_ToMean/version_1/ckpt.pth'
                # ckpt = torch.load(ckpt_path)
                # model[k].load_state_dict(ckpt)
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
    def __init__(self, model,config,test_loader, model_type = 'RN50'):
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
        
        self.kl_weight  = 0.0001
        self.z_dim = self.config['models']['eeg']['params']['z_dim']

        pretrain_map={
                'RN50':'openai',
                'RN101':'openai',
                'RN50x4':'openai',
                'ViT-B-16':'laion2b_s34b_b88k',
                'ViT-B-32':'laion2b_s34b_b79k',
                'ViT-L-14':'laion2b_s32b_b82k',
                'ViT-H-14':'laion2b_s32b_b79k',
                'ViT-g-14':'laion2b_s34b_b88k', 
                'ViT-bigG-14':'laion2b_s39b_b160k',
            }
        self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(model_type, device="cuda:2",pretrained=pretrain_map[model_type])
        self.vlmodel.eval()

    def forward(self, batch,sample_posterior=False):
        eeg, label, img_path,img, img_features, text, text_features, session, subject,eeg_mean = batch
        
        # posterior = self.eeg.encode(eeg , scale)
        # if sample_posterior:
        #     eeg_latent = posterior.sample()
        # else:
        #     eeg_latent = posterior.mode()
        # eeg_latent = self.eeg.encoder(eeg)
        eeg_latent = self.eeg(eeg)
        debia_latent = self.debias(eeg_latent[:,:self.z_dim])

        logit_scale = self.debias.logit_scale
        clip_loss = self.criterion(debia_latent, img_features, logit_scale)
        
        # kl_loss = torch.sum(posterior.kl())/ eeg.shape[0]

        loss = clip_loss #+ self.kl_weight * kl_loss
        return eeg_latent, debia_latent, img_features, loss
    
    def training_step(self, batch, batch_idx):
        eeg, label, img_path,img, img_features, text, text_features, session, subject, eeg_mean = batch
        eeg_latent, debia_latent, img_features, loss = self(batch,sample_posterior=True)

        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
     
    def validation_step(self, batch, batch_idx):
        self.all_image_features = self.all_image_features.to(self.device)
    
        eeg, label, img_path,img, img_features, text, text_features, session, subject, eeg_mean = batch
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
        eeg, label, img_path,img, img_features, text, text_features, session, subject, eeg_mean = batch
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
        #freeze eeg
        # for param in self.eeg.parameters():
        #     param.requires_grad = False

        optimizer = globals()[self.config['train']['optimizer']](self.parameters(), lr = self.config['train']['lr'])

        # scheduler = {
        #     'scheduler': lr_scheduler.SequentialLR(
        #         optimizer,
        #         schedulers=[
        #             lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=5),
        #             lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-4)
        #         ],
        #         milestones=[5]
        #     ),
        #     'interval': 'epoch', 
        #     'frequency': 1,
        #     'monitor': 'val_loss'
        # }
        return [optimizer]#, [scheduler]
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/wht/multimodal_brain/src/tasks/base/configs/debias_train.yaml",
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

    parser.add_argument(
        "--ksize",
        default='[10,15]',
        type=str,
    )

    parser.add_argument(
        "--sigmaX",
        default='[4, 9]',
        type=str,
    )


    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]
    config['ksize'] = eval(config['ksize'])
    config['sigmaX'] = eval(config['sigmaX'])
    print(config)

    os.makedirs(config['save_dir'],exist_ok=True)
    logger = TensorBoardLogger(config['save_dir'], name="saliency", version=f"{config['data']['subjects']}_{config['ksize']}_{config['sigmaX']}")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, test_loader = load_data(config)
    
    pl_model = load_model(config,test_loader)

    checkpoint_callback = ModelCheckpoint(
            monitor='val_top1_acc',
            mode='max',
            save_top_k=3)

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'], devices=[2],accelerator='cuda',logger=logger)
    print(trainer.logger.log_dir, trainer.logger.version)
    ckpt_path = "/home/wht/multimodal_brain/src/tasks/exp/VAE_finetune/version_4/checkpoints/epoch=99-step=1700.ckpt"
    ckpt_path = None
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader,ckpt_path=ckpt_path)
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    with open(os.path.join(logger.log_dir,'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)
     
if __name__=="__main__":
    main()