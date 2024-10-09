import torch,os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from base.u2net import U2NET
from base.utils import RescaleT,ToTensorLab


class RandomCircularBlur:
    def __init__(self):
        pass

    def __call__(self, img):
        # 随机选择模糊的核大小
        self.blur_kernel_size_outer = random.randint(25, 35)  # 外部大模糊
        self.blur_kernel_size_middle = random.randint(5, 15)  # 中间小模糊

        # 模糊核大小必须是奇数
        if self.blur_kernel_size_outer % 2 == 0:
            self.blur_kernel_size_outer += 1
        if self.blur_kernel_size_middle % 2 == 0:
            self.blur_kernel_size_middle += 1

        # 定义模糊区域的比例
        self.middle_radius_ratio = np.clip(np.random.normal(0.5, 0.05), 0.4, 0.6)  # 中间圆的比例
        self.outer_radius_ratio = np.clip(np.random.normal(0.75, 0.05), 0.6, 0.8)  # 外部区域的比例

        # 将图像转换为 PIL 格式，如果需要
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        # 将图像转换为 NumPy 数组
        img_np = np.array(img)
        height, width, _ = img_np.shape

        # 计算不同区域的半径
        middle_radius = int(min(width, height) * self.middle_radius_ratio//2)
        outer_radius = int(min(width, height) * self.outer_radius_ratio//2)
        center_x = width // 2
        center_y = height // 2

        # 创建遮罩
        mask = np.zeros((height, width), dtype=np.uint8)

        # 中间圆保持原始图像
        cv2.circle(mask, (center_x, center_y), middle_radius, 255, -1)

        # 中间圆环轻微模糊
        mask_middle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask_middle, (center_x, center_y), outer_radius, 255, -1)
        cv2.circle(mask_middle, (center_x, center_y), middle_radius, 0, -1)

        # 外部区域更强的模糊
        mask_outer = np.ones((height, width), dtype=np.uint8) * 255
        cv2.circle(mask_outer, (center_x, center_y), outer_radius, 0, -1)

        # 处理图像模糊
        img_blur_outer = cv2.GaussianBlur(img_np, (self.blur_kernel_size_outer, self.blur_kernel_size_outer), 0)
        img_blur_middle = cv2.GaussianBlur(img_np, (self.blur_kernel_size_middle, self.blur_kernel_size_middle), 0)

        # 合成最终图像
        img_result = img_np.copy()
        img_result[mask == 255] = img_np[mask == 255]  # 中心圆保持不变
        img_result[mask_middle == 255] = img_blur_middle[mask_middle == 255]  # 中间圆环轻微模糊
        img_result[mask_outer == 255] = img_blur_outer[mask_outer == 255]  # 外部区域强模糊

        return img_result

class CompleteBlur:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        # samples = np.random.normal(loc=40, scale=0.1)
        # int_samples = np.round(samples).astype(int)
        # odd_samples = int_samples + (int_samples % 2 == 0)
        # odd_samples = 41
        # self.blur_kernel_size = odd_samples
        img_np = np.array(img)
        img_blur_outer = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        return img_blur_outer

class RandomBlur:
    def __init__(self ):
        pass
    def __call__(self, img):

        samples = np.random.normal(loc=40, scale=0.1)
        int_samples = np.round(samples).astype(int)
        odd_samples = int_samples + (int_samples % 2 == 0)
        self.blur_kernel_size = odd_samples
        img_np = np.array(img)
        img_blur_outer = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        return img_blur_outer
    
class ScaleCircularBlur:
    def __init__(self):
        pass

    def __call__(self, img):
        self.blur_kernel_size_middle = 25 #30
        self.blur_kernel_size_outer = 80 #30
        

        if self.blur_kernel_size_outer % 2 == 0:
            self.blur_kernel_size_outer += 1
        if self.blur_kernel_size_middle % 2 == 0:
            self.blur_kernel_size_middle += 1

        self.middle_radius_ratio =0.35 #0.2
        self.outer_radius_ratio =0.8 #0.7

        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)

        img_np = np.array(img)
        height, width, _ = img_np.shape

        middle_radius = int(min(width, height) * self.middle_radius_ratio//2)
        outer_radius = int(min(width, height) * self.outer_radius_ratio//2)
        center_x = width // 2
        center_y = height // 2

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), middle_radius, 255, -1)

        mask_middle = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask_middle, (center_x, center_y), outer_radius, 255, -1)
        cv2.circle(mask_middle, (center_x, center_y), middle_radius, 0, -1)

        img_blur_outer = cv2.GaussianBlur(img_np, (self.blur_kernel_size_outer, self.blur_kernel_size_outer), 0)
        img_blur_middle = cv2.GaussianBlur(img_np, (self.blur_kernel_size_middle, self.blur_kernel_size_middle), 0)
        img_blur_center = cv2.GaussianBlur(img_np, (5, 5), 0)

        img_result = img_blur_outer.copy()
        img_result[mask == 255] = img_blur_center[mask == 255]
        img_result[mask_middle == 255] = img_blur_middle[mask_middle == 255]
        # img_result[mask_outer == 255] = img_blur_outer[mask_outer == 255]

        return img_result
    
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

class DynamicBlur:
    def __init__(self, ksize = [21,21] , sigmaX = [1, 2]):
        if ksize[0] % 2 == 0:
            ksize[0] += 1
        if ksize[1] % 2 == 0:
            ksize[1] += 1
        self.ksize = ksize
        self.sigmaX = sigmaX
        self.model = U2NET(3,1)
        model_path = '/home/wht/multimodal_brain/src/tasks/exp/u2net/u2net.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()
        self.T= transforms.Compose([RescaleT(320),ToTensorLab(flag=0)])

    def __call__(self, img):
        img = np.array(img)
        src_img = img.copy()
        img =  self.T(img).unsqueeze(0)
        img = img.cuda()
        d1,d2,d3,d4,d5,d6,d7= self.model(img)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np*255).convert('RGB')

        imo = im.resize((src_img.shape[1],src_img.shape[0]),resample=Image.BILINEAR)
        pb_np = np.array(imo)
        binary_matrix = np.where(pb_np[:,:,0] > 127, 1, 0)

        img_blur_center = cv2.GaussianBlur(src_img, (self.ksize[0], self.ksize[0]), self.sigmaX[0])
        img_blur = cv2.GaussianBlur(src_img, (self.ksize[1], self.ksize[1]), self.sigmaX[1])
        
        img_result = img_blur.copy()
        img_result[binary_matrix == 1] = img_blur_center[binary_matrix == 1] #center

        return img_result


class Saliency:
    def __init__(self, ksize = [21,21] , sigmaX = [1, 2]):
        
        self.model = U2NET(3,1)
        model_path = '/home/wht/multimodal_brain/src/tasks/exp/u2net/u2net.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.cuda()
        self.model.eval()
        self.T= transforms.Compose([RescaleT(320),ToTensorLab(flag=0)])

    def __call__(self, img):
        img = np.array(img)
        src_img = img.copy()
        img =  self.T(img).unsqueeze(0)
        img = img.cuda()
        d1,d2,d3,d4,d5,d6,d7= self.model(img)
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        predict = pred
        predict = predict.squeeze()
        predict_np = predict.cpu().data.numpy()
        im = Image.fromarray(predict_np*255).convert('RGB')

        imo = im.resize((src_img.shape[1],src_img.shape[0]),resample=Image.BILINEAR)
        pb_np = np.array(imo)
        binary_matrix = np.where(pb_np[:,:,0] > 127, 1, 0)

        height, width = src_img.shape[:2]
        white_img = np.ones((height, width, 3), dtype=np.uint8) * 255

        img_result = white_img
        img_result[binary_matrix == 0] = src_img[binary_matrix == 0] #center

        return img_result
   
class InpaintingDataset():
    def __init__(self, data_dir='/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_1000Hz_whiten', mode='train',model_type = 'ViT-B-32',transform=None):
        self.data_dir = data_dir
        self.mode = mode 
        if transform == None:
            self.transform =  transforms.Compose([
                            RandomCircularBlur(),
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),  
                            # transforms.RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
                        ])
        else:
            self.transform = transform
        self.img_directory = os.path.join(self.data_dir,'../','Image_set',f'{mode}_images')
        self.images = []
        all_folders = [d for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        all_folders.sort()
        for i,folder in enumerate(all_folders):
            folder_path = os.path.join(self.img_directory, folder)
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            self.images.extend(os.path.join(folder_path, img).rsplit("Image_set/")[-1] for img in all_images)
        
        self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        self.all_class_names.sort()

        self.n_cls = 1654 if self.mode=='train' else 200
        
        features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}.pt")
        if os.path.exists(features_filename) :
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            pretrain_map={
                'RN50':'openai',
                'RN101':'openai',
                'RN50x4':'openai',
                'ViT-B-16':'laion2b_s34b_b88k',
                'ViT-B-32':'laion2b_s34b_b79k',
                'ViT-L-14':'laion2b_s32b_b82k',
                'ViT-H-14':'laion2b_s32b_b79k',
                'ViT-g-14':'laion2b_s34b_b88k', 
                'ViT-bigG-14':'laion2b_s39b_b160k'}
            
            self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(model_type, device="cuda",pretrained=pretrain_map[model_type]) #laion2b_s39b_b160k
  
            self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, features_filename)
        
        if mode =='test':
            self.all_text_features = torch.from_numpy(np.concatenate([self.text_features[k].unsqueeze(0) for k in self.all_class_names]))
            self.all_image_features = torch.from_numpy(np.concatenate([self.img_features[k].unsqueeze(0) for k in self.img_features]))
    
    @torch.no_grad()
    def ImageEncoder(self,images):
        set_images = list(set(images))
        set_images.sort()
        batch_size = 64
        image_features_list = []
        for i in range(0, len(set_images), batch_size):
            batch_images = set_images[i:i + batch_size]
            image_inputs = torch.stack([self.preprocess(Image.open(os.path.join(self.data_dir,'../Image_set',img)).convert("RGB")) for img in batch_images])
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]:image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text])
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict
    
    def __getitem__(self, index):
        img_path = self.images[index]
        
        src_img = Image.open(os.path.join(self.img_directory,'../',img_path)).convert("RGB")
        transform = transforms.ToTensor()
        src_img_tensor = transform(src_img)
        src_img_latent = self.img_features[img_path]

        attention_img = self.transform(src_img)
        return src_img_tensor, attention_img, src_img_latent

    def __len__(self):
        return len(self.images)