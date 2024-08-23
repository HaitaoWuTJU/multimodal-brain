import os
# from diffusers import DiffusionPipeline, StableDiffusionPipeline,StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
import torch
from pytorch_lightning import seed_everything


from _pipeline import StableDiffusionXLPipeline, retrieve_timesteps
from models import Autoencoder
seed_everything(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from omegaconf import OmegaConf
config = OmegaConf.load("/root/workspace/wht/multimodal_brain/src/tasks/img_generation/configs/base.yaml")

model = {}
for k,v in config['models'].items():
    if k == 'generation':
        model[k] = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
    else:
        model[k] = globals()[v['name']](**v['args'])
        
sd_model = model['generation']


prompt = "a photo of a cat on mars"
# prompt = ""
n_steps = 30
high_noise_frac = 1.0
output = sd_model(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    # output_type="latent",
)

output['images'][0].save('results/mar_cat_source.png')