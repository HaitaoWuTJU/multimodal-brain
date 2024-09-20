import pytorch_lightning as pl

from torch.optim import AdamW, Adam
import torchmetrics,os
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import io, torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from omegaconf import OmegaConf

from base.models import Autoencoder
from base.autoencoder import AutoencoderKL
from base.utils import instantiate_from_config

def load_model(config,test_loader):
    # config = '/home/wht/multimodal_brain/src/tasks/base/configs/autoencoder_kl_32x32x4.yaml'
    # config = OmegaConf.load(config)
    # model = instantiate_from_config(config['model'])
    model = instantiate_from_config(config['models']['eeg'])
    return model