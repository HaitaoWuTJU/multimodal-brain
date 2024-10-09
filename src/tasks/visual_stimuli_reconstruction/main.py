from base.utils import update_config, SoftClipLoss, ClipLoss
from base.utils import instantiate_from_config
from base._pipeline import StableDiffusionXLPipeline
from base.data import load_data
import argparse
import os
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
from torch.nn import MSELoss
# from lavis.models.clip_models.loss import ClipLoss

# import user lib
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

# user env
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'


class PLModel(pl.LightningModule):
    def __init__(self, config, test_loader, model_type='RN50'):
        super().__init__()

        self.config = config

        # init model
        for k, v in config['models'].items():
            model = instantiate_from_config(v)
            setattr(self, f"{k}", model)

        self.criterion = MSELoss()

    def forward(self, batch):
        eeg, label, img_path, img, img_z, text, text_features, session, subject, eeg_mean = batch

        eeg_z = self.eeg(eeg)
        eeg_z = eeg_z.view(eeg.shape[0], 4, 62, 62)

        clip_loss = self.criterion(eeg_z, img_z)

        loss = clip_loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss

    def on_test_epoch_end(self):
        avg_test_loss = self.trainer.callback_metrics['test_loss']
        return {'test_loss': avg_test_loss.item()}

    def configure_optimizers(self):
        optimizer = globals()[self.config['train']['optimizer']](
            self.parameters(), lr=self.config['train']['lr'])
        return [optimizer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/wht/multimodal_brain/src/tasks/visual_stimuli_reconstruction/config.yaml",
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

    logger = TensorBoardLogger(config['save_dir'], name='ldm', version=f"{config['data']['subjects']}")
    os.makedirs(logger.log_dir, exist_ok=True)
    
    shutil.copy(opt.config, os.path.join(
        logger.log_dir, opt.config.rsplit('/', 1)[-1]))

    train_loader, test_loader = load_data(config)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_top1_acc',
        mode='max',
        save_top_k=3)

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=True), callbacks=[
                      checkpoint_callback], max_epochs=config['train']['epoch'], devices=[2], accelerator='cuda', logger=logger)
    print(trainer.logger.log_dir, trainer.logger.version)
    ckpt_path = "/home/wht/multimodal_brain/src/tasks/exp/VAE_finetune/version_4/checkpoints/epoch=99-step=1700.ckpt"
    ckpt_path = None
    trainer.fit(pl_model, train_dataloaders=train_loader,
                val_dataloaders=test_loader, ckpt_path=ckpt_path)
    test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
    with open(os.path.join(logger.log_dir, 'test_results.json'), 'w') as f:
        json.dump(test_results, f, indent=4)


if __name__ == "__main__":
    main()
