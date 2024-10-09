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
from .inpating_data import ScaleCircularBlur,CompleteBlur, DynamicBlur, RandomBlur, Saliency
from tqdm import tqdm

def load_data(config):
    data_config = config['data']

    test_dataset = EEGDataset(data_dir=data_config['data_dir'],subjects=data_config['subjects'],model_type=data_config['model_type'],mode='test',selected_ch=data_config['selected_ch'],transform=None,avg=data_config['test_avg'],GaussianBlur=data_config['GaussianBlur'],timesteps=config['models']['eeg']['params']['timesteps'],\
                              blur_kernel_size=data_config['blur_kernel_size'], ksize = config['ksize'],sigmaX = config['sigmaX'])
    print('init test_dataset success')
    train_dataset = EEGDataset(data_dir=data_config['data_dir'],subjects=data_config['subjects'],model_type=data_config['model_type'],mode='train',selected_ch=data_config['selected_ch'],transform=None,avg=data_config['train_avg'],GaussianBlur=data_config['GaussianBlur'],timesteps=config['models']['eeg']['params']['timesteps'],\
                               blur_kernel_size=data_config['blur_kernel_size'],  ksize = config['ksize'],sigmaX = config['sigmaX'])
    print('init train_dataset success')
    # train_size = len(train_dataset) * data_config['train_val_rate']
    # val_size = len(train_dataset) - train_size
    # train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    test_loader = DataLoader(test_dataset, batch_size=data_config['test_batch_size'], shuffle=False, drop_last=False,num_workers=25, pin_memory=True)
    # val_loader = DataLoader(val_subset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=data_config['train_batch_size'], shuffle=True, drop_last=False, num_workers=64, pin_memory=True)

    return train_loader, test_loader


class EEGDataset():
    def __init__(self, data_dir, subjects, mode, model_type = 'ViT-B-32',selected_ch=None,transform=None,avg=False,GaussianBlur=False,timesteps=800,blur_kernel_size=21,ksize=[21,31],sigmaX=[1,2]):
        self.data_dir = data_dir
        self.img_directory = os.path.join(self.data_dir,'../','Image_set',f'{mode}_images')
        self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        self.all_class_names.sort()
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels
        self.transform = transform
        self.avg = avg
        self.GaussianBlur = GaussianBlur
        self.timesteps = timesteps
        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 4 if self.mode=='train' else 80
        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)
        
        self.blur_kernel_size = blur_kernel_size
        if self.GaussianBlur == 'avg':
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}_avg_blur_{self.blur_kernel_size}.pt") #complete scale
            T = [CompleteBlur(self.blur_kernel_size),transforms.ToPILImage()]
        
        elif self.GaussianBlur == 'scale':
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}_scale_blur.pt") #complete scale
            T = [ScaleCircularBlur(),transforms.ToPILImage()]

        elif self.GaussianBlur == 'dynamic':
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}_dynamic_blur_{ksize}_{sigmaX}.pt") #complete scale
            T = [DynamicBlur(ksize,sigmaX),transforms.ToPILImage()]
        elif self.GaussianBlur == 'random':
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}_dynamic_blur_random.pt")
            T = [RandomBlur(),transforms.ToPILImage()]
        elif self.GaussianBlur == 'saliency':
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}_saliency_reverse.pt")
            T =[Saliency(),transforms.ToPILImage()]
        else:
            features_filename = os.path.join(self.data_dir,'../Image_feature',f"{model_type.replace('/','-')}_features_{mode}.pt")
            T = []

        pretrain_map= {
                'RN50':{'pretrained':'openai','resize':(224,224)}, #1024 
                'RN101':{'pretrained':'openai','resize':(224,224)}, #512
                'RN50x4':{'pretrained':'openai','resize':(288,288)}, #640
                'RN50x16':{'pretrained':'openai','resize':(384,384)}, #768
                'RN50x64':{'pretrained':'openai','resize':(448,448)},  #1024
                'ViT-B-16':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #512
                'ViT-B-32':{'pretrained':'laion2b_s34b_b79k','resize':(224,224)}, #512
                'ViT-L-14':{'pretrained':'laion2b_s32b_b82k','resize':(224,224)}, #768
                'ViT-H-14':{'pretrained':'laion2b_s32b_b79k','resize':(224,224)}, #1024
                'ViT-g-14':{'pretrained':'laion2b_s34b_b88k','resize':(224,224)}, #1024
                'ViT-bigG-14':{'pretrained':'laion2b_s39b_b160k','resize':(224,224)}, #1280
            }
     
        T.extend([transforms.Resize(pretrain_map[model_type]['resize']),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
        self.transform =  transforms.Compose(T)
        # features_filename = os.path.join(self.data_dir,'../Image_feature/../',f"{model_type.replace('/','-')}_features_{mode}.pt")
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(model_type, device="cuda:0",pretrained=pretrain_map[model_type]['pretrained']) #laion2b_s39b_b160k
            self.vlmodel.eval()
            self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, features_filename)
        
        
        if mode =='test':
            self.all_text_features = torch.from_numpy(np.concatenate([self.text_features[k].unsqueeze(0) for k in self.all_class_names]))
            self.all_image_features = torch.from_numpy(np.concatenate([self.img_features[k].unsqueeze(0) for k in self.img_features]))
    
    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('250HZ',1)[-1]}----")
        loaded_data = torch.load(data_path)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:,0]
            avg_data['img'] = loaded_data['img'][:,0]
            avg_data['text'] = loaded_data['text'][:,0]
                
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    @torch.no_grad()
    def ImageEncoder(self,images):
        
        set_images = list(set(images))
        set_images.sort()
        batch_size = 64
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]
            # image_inputs = torch.stack([self.preprocess(Image.open(os.path.join(self.data_dir,'../Image_set',img)).convert("RGB")) for img in batch_images])

            device = next(self.vlmodel.parameters()).device

            ele = [self.transform(Image.open(os.path.join(self.data_dir,'../Image_set',img)).convert("RGB")) for img in batch_images]
            image_inputs = torch.stack(ele).to(device)

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
        device = next(self.vlmodel.parameters()).device
        text_inputs =  text_inputs.to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index//self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        # img = Image.open(os.path.join(self.data_dir,'../Image_set',img_path)).convert("RGB")
        # img = self.transform(img)
        img = 'None'

        img_features = self.img_features[img_path]
        text =  f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]
        
        return eeg[:,:self.timesteps], label, img_path,img, img_features, text, text_features,session,subject,eeg_mean[:,:self.timesteps]

    def __len__(self):
        return self.trial_all_subjects
    

class EEGDatasetLDM():
    def __init__(self, data_dir, subjects, mode, selected_ch=None,transform=None,avg=False,GaussianBlur=False,timesteps=800):
        self.data_dir = data_dir
        self.img_directory = os.path.join(self.data_dir,'../','Image_set',f'{mode}_images')
        self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        self.all_class_names.sort()
        self.subjects = subjects
        self.mode = mode
        self.selected_ch = selected_ch
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']
        if self.selected_ch == "None":
            self.selected_ch = self.channels
        self.transform = transform
        self.avg = avg
        self.GaussianBlur = GaussianBlur
        self.timesteps = timesteps
        self.n_cls = 1654 if self.mode=='train' else 200
        self.per_trials = 4 if self.mode=='train' else 80
        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)

        features_filename = '/home/wht/multimodal_brain/datasets/things-eeg-small/Image_feature/ldm_train.pt'
        if os.path.exists(features_filename):
            saved_features = torch.load(features_filename)
            self.img_z = saved_features['ldm_z']
    
    def load_data(self,data_path):
        logging.info(f"----load {data_path.rsplit('250HZ',1)[-1]}----")
        loaded_data = torch.load(data_path)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:,0]
            avg_data['img'] = loaded_data['img'][:,0]
            avg_data['text'] = loaded_data['text'][:,0]
                
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1,*loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data
        
        
        for k,v in loaded_data.items():
            if k in ['eeg','label','img','text','session']:
                logging.info(f"{k}: {v.shape}")
        return loaded_data    
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index//self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        img = Image.open(os.path.join(self.data_dir,'../Image_set',img_path)).convert("RGB")

        img_z = self.img_z[img_path]
        text =  f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]
        
        return eeg[:,:self.timesteps], label, img_path, img, img_z, text, text_features,session,subject,eeg_mean[:,:self.timesteps]

    def __len__(self):
        return self.trial_all_subjects