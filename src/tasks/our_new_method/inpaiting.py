import argparse, os, sys
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from pytorch_lightning.strategies import DDPStrategy
import open_clip
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.inpating_data import InpaintingDataset
from base.models import ProjectLayer

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

class PLModel(pl.LightningModule):
    def __init__(self,model,model_type='ViT-B-32'):
        super().__init__()

        self.model = model
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
        self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(model_type, device="cuda",pretrained=pretrain_map[model_type])
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        src_img, attention_img, src_img_latent = batch
        attention_img_latent = self.vlmodel.encode_image(attention_img)
        recon_img_latent = self.model(attention_img_latent)
        loss = self.criterion(recon_img_latent,src_img_latent)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True,prog_bar=True, logger=True, sync_dist=True, batch_size=batch[0].shape[0])
        return loss
    
    def configure_optimizers(self):
        for param in self.vlmodel.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        return {'optimizer': optimizer}
    
if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()
    seed_everything(opt.seed)

    logger = TensorBoardLogger('/home/wht/multimodal_brain/src/tasks/exp', name="inpainting")
    os.makedirs(logger.log_dir,exist_ok=True)

    test_dataset = InpaintingDataset(mode='test')
    train_dataset = InpaintingDataset(mode='train')

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False,num_workers=32)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False, num_workers=32, pin_memory=True)

    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=3)

    trainer = Trainer(log_every_n_steps=10, strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=100, devices=[5],accelerator='cuda',logger=logger)
    print(trainer.logger.log_dir, trainer.logger.version)
    ckpt_path = "/home/wht/multimodal_brain/src/tasks/exp/VAE_finetune/version_4/checkpoints/epoch=99-step=1700.ckpt"
    ckpt_path = None

    pl_model = PLModel(model=ProjectLayer(embedding_dim=512, proj_dim=512))
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader,ckpt_path=ckpt_path)