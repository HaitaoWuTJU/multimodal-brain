import argparse, os
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from diffusers import DiffusionPipeline
from pytorch_lightning.strategies import DDPStrategy


##import user lib
from models import load_model
from data import load_data

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
        default="configs/base.yaml",
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
    
    # eeg_model = load_model(config)['eeg']
    # model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    # pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path=model_name, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    # pl_model = StableDiffusionFineTuner(pipeline,eeg_model)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1,
    )
    # checkpoint_path = "/root/workspace/wht/multimodal_brain/src/tasks/img_generation/lightning_logs/version_33/checkpoints/epoch=4-step=12925.ckpt"
    checkpoint_path =  None
    trainer = Trainer(strategy=DDPStrategy(find_unused_parameters=True),callbacks=[checkpoint_callback],max_epochs=config['train']['epoch'],  devices=1, accelerator='gpu')
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader,ckpt_path=checkpoint_path)
    trainer.test(pl_model, test_loader)

if __name__=="__main__":
    main()