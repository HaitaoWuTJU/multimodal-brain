import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from diffusers import DiffusionPipeline
from pytorch_lightning.strategies import DDPStrategy
import sys

##import user lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from pl_model import load_model
from base.data import load_data

#user env
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/home/wht/multimodal_brain/src/tasks/base/configs/generation.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    print(config)

    train_loader, test_loader = load_data(config)
    
    pl_model = load_model(config,test_loader)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
    )

    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'],  devices=1, accelerator='gpu')
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    trainer.test(pl_model, test_loader)

if __name__=="__main__":
    main()