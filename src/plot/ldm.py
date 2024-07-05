import torch
from torchvision.transforms import ToPILImage
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter, Decoder, DALLE2
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig, TrainDecoderConfig


# prior_config = TrainDiffusionPriorConfig.from_json_path("/root/workspace/wht/pretrained/dalle2/prior/prior_config.json").prior
# prior = prior_config.create().cuda()

# prior_model_state = torch.load("/root/workspace/wht/pretrained/dalle2/prior/latest.pth")
# prior.load_state_dict(prior_model_state, strict=True)

decoder_config = TrainDecoderConfig.from_json_path("/root/workspace/wht/pretrained/dalle2/decoder_config.json").decoder
decoder = decoder_config.create().cuda()

decoder_model_state = torch.load("/root/workspace/wht/pretrained/dalle2/decoder.pth")

for k in decoder.clip.state_dict().keys():
    decoder_model_state["clip." + k] = decoder.clip.state_dict()[k]

decoder.load_state_dict(decoder_model_state, strict=True)

print(1)
# dalle2 = DALLE2(prior=prior, decoder=decoder).cuda()

# images = dalle2(
#     ['.'],
#     cond_scale = 2.
# ).cpu()

# print(images.shape)

# for img in images:
#     img = ToPILImage()(img)
#     img.show()