import torch,sys,os
from omegaconf import OmegaConf
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from base.utils import instantiate_from_config
path = '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_ToMean/version_1/checkpoints/epoch=99-step=4100.ckpt'
ckpt = torch.load(path)

config = OmegaConf.load("/home/wht/multimodal_brain/src/tasks/base/configs/pretrain.yaml")
model = instantiate_from_config(config['models']['eeg'])

model.load_state_dict(ckpt['state_dict'])

torch.save(model.state_dict(), '/home/wht/multimodal_brain/src/tasks/exp/VAE_Pretrain_ToMean/version_1/ckpt.pth')