import torch
from torch import autocast
# from pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
    
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['http_proxy'] = 'http://127.0.0.1:7890'

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

def main():

    import torch

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16)
    pipe.to("cuda")

    # get image
    url = "cat.png"
    image = load_image(url)

    # run image variation
    image = pipe(image,noise_level=0).images[0]
    
    image.save("cat_sd_unclip.png")

         
if __name__=="__main__":
    main()