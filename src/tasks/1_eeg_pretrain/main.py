import argparse, os,sys
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from diffusers import DiffusionPipeline
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
##import user lib
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
        default="/home/wht/multimodal_brain/src/tasks/base/configs/pretrain.yaml",
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
    
    os.makedirs(config['save_dir'],exist_ok=True)
    
    logger = TensorBoardLogger(config['save_dir'], name="VAE Pretrain")
    os.makedirs(logger.log_dir,exist_ok=True)
    shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

    train_loader, test_loader = load_data(config)
    
    pl_model = load_model(config,test_loader)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
    )

    ckpt_path = config.get('ckpt_path', None)
    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'], devices=[6],accelerator='cuda',logger=logger)
   
    print(trainer.logger.save_dir, trainer.logger.version)
    
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

if __name__=="__main__":
    main()