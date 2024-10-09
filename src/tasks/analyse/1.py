import sys,os,torch
import pandas as pd
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from base.utils import instantiate_from_config
from base.data import load_data
import numpy as np
import pandas as pd
import seaborn as sns
torch.cuda.empty_cache()

config = '/home/wht/multimodal_brain/src/tasks/base/configs/autoencoder_kl_32x32x4.yaml'
config = OmegaConf.load(config)
model = instantiate_from_config(config['model'])
model = model.to(device)

ckpt_pth = '/home/wht/multimodal_brain/src/tasks/1_eeg_pretrain/exp/VAE Pretrain/version_29/checkpoints/epoch=8-step=4257.ckpt'
ckpt= torch.load(ckpt_pth)


# model.load_state_dict(ckpt['state_dict'])

data_path= '/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten/sub-01/test.pt'
loaded_data = torch.load(data_path)
loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])