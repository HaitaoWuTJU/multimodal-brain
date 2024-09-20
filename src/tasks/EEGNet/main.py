import argparse, os,sys
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.accelerators import find_usable_cuda_devices

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
        default="/home/wht/multimodal_brain/src/tasks/base/configs/eegnet.yaml",
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
    train_loader, test_loader = load_data(config)
    dataset = train_loader.dataset
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):

        logger = TensorBoardLogger(config['save_dir'], name="eegnet",version=f"1_f{fold}")
        os.makedirs(logger.log_dir,exist_ok=True)
        shutil.copy(opt.config, os.path.join(logger.log_dir,opt.config.rsplit('/',1)[-1]))

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader_fold = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_subset, batch_size=train_loader.batch_size, shuffle=False)

        pl_model = load_model(config,test_loader)
        
        checkpoint_callback = ModelCheckpoint(
            monitor='val_top1_acc',
            mode='max',
            save_top_k=-1,
        )
        trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=False),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'], devices=[6],accelerator='gpu', logger=logger)
    
        print(trainer.logger.save_dir, trainer.logger.version)
        
        trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)

        trainer.test(ckpt_path='best', dataloaders=test_loader)
        break
if __name__=="__main__":
    main()