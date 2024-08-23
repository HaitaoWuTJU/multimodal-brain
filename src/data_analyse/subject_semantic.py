import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import time
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import umap
from lavis.models.clip_models.loss import ClipLoss
from torch.utils.data import DataLoader, Dataset
import random
import itertools
from sklearn.metrics import accuracy_score
import argparse
import ast

import sys
sys.path.append('/root/workspace/wht/multimodal_brain/src')
from models.mlp import MLP,ProjectLayer,Direct
from models.eeg import EEGEncoder,LSTMModel
from models.ae import Autoencoder
from dataset.things_eeg import EEGDataset

from utils import set_seed,update_config,set_logging


set_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
model_type = 'ViT-B-32'
latend_dim_dict = {'ViT-B-16':512,'ViT-B-32':512,'ViT-L-14': 768,'RN50':1024,'RN101':512,'RN50x4':640,'ViT-H-14':1024,'ViT-g-14':1024,'ViT-bigG-14':1280}
latend_dim = latend_dim_dict[model_type]
config = {
    "data_dir": "/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten",#"/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz/",
    "exp_root":'./exp',
    "device":device,
    "name": 'train_model_plot_subject',
    "lr": 1e-4,
    "epochs": 24,
    "batch_size": 256,
    "model_type":model_type,
    "latend_dim":latend_dim,
    "logger": True,
    "subjects":['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
    # "eeg":{'name':'ProjectLayer','args':{'embedding_dim':len(selected_ch)*250, 'proj_dim':latend_dim}},
    
    "model":{"eeg":{'name':'Autoencoder','args':{'in_chans':len(selected_ch)}},
                "eeg_semantic":{'name':'ProjectLayer','args':{'embedding_dim':latend_dim, 'proj_dim':latend_dim}},
                "aux":{'name':'MLP','args':{'input_dim':latend_dim_dict[model_type],'output_dim':10,'hiden_dims':[]}}},
    # "eeg":{'name':'Autoencoder','args':{'in_chans':len(selected_ch)}},
    # "aux":{'name':'MLP','args':{'input_dim':latend_dim_dict[model_type],'output_dim':10,'hiden_dims':[]}},
    # "aux2":{'name':'MLP','args':{'input_dim':latend_dim_dict[model_type],'output_dim':10,'hiden_dims':[]}},
}
config['exp_dir'] = os.path.join(config['exp_root'],config['name'])

os.makedirs(config['exp_dir'],exist_ok=True)

set_logging(os.path.join(config['exp_dir'],f"{'_'.join(config['subjects'])}.LOG"))
logging.info(f"-------------------------START-------------------------")
logging.info(f"CONFIG: {config}")

transform = transforms.Compose([
])

test_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='test',selected_ch=selected_ch,transform=transform,avg=True)
train_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='train',selected_ch=selected_ch,transform=transform,avg=False)
    
logging.info(f"train num: {len(train_dataset)}, test num: {len(test_dataset)}")
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

data={
    'train':train_loader,
    'test':test_loader
}
model = {k:globals()[v['name']](**v['args']).to(device) for k,v in config["model"].items()}

train_eval_args = {
    'optimizer':optim.Adam([{'params': v.parameters(), 'lr': config['lr']} for k,v in model.items()]),
    'criterion':ClipLoss(),
    'epochs':config['epochs'],
    'subjects':config['subjects'],
    'best_acc':-1,
    'best_epoch':-1,
    'best_train_acc':-1,
}

exp_dir = 'SoftClipLoss_gan_transformer_lr1e-4_bs256'
for epoch in range(25):
    ckpt = torch.load(os.path.join('..',config['exp_root'],exp_dir,f'ckpt_{epoch}_sub-01_sub-02_sub-03_sub-04_sub-05_sub-06_sub-07_sub-08_sub-09_sub-10.pth'),map_location=config['device'])
    for k,v in model.items():
        v.load_state_dict(ckpt[k])
        v.eval()

    X = []
    Y = []
    S = []
    Source_X = []
    EEG =[]

    all_predicted_classes = []
    all_true_labels = []

    all_text_features=data['test'].dataset.all_text_features
    all_image_features=data['test'].dataset.all_image_features
    all_text_features = all_text_features/all_text_features.norm(dim=-1, keepdim=True).to(device)
    all_image_features = all_image_features/all_image_features.norm(dim=-1, keepdim=True).to(device)
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features ,session,subject = sample
        eeg = eeg.to(device)
        label = label.to(device)
        
        img_features = img_features.to(device)
        text_features = text_features.to(device)
        
        eeg_features = model['eeg'].forward_cls(eeg)
        eeg_semantic = model['eeg_semantic'](eeg_features)
        

        eeg_semantic = eeg_semantic/eeg_semantic.norm(dim=-1, keepdim=True)
        img_features = img_features/img_features.norm(dim=-1, keepdim=True)
    
        
        Source_X.append(eeg.detach().cpu().numpy())
        EEG.append(eeg_features.detach().cpu().numpy())
        X.append(eeg_semantic.detach().cpu().numpy())
        Y.append(img_features.detach().cpu().numpy())
        S.append(subject.detach().cpu().numpy())
        
        similarity = (eeg_semantic @ all_image_features.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        all_predicted_classes.append(top_k_indices.cpu().numpy())
        all_true_labels.extend(label.cpu().numpy())
    all_predicted_classes = np.concatenate(all_predicted_classes,axis=0)
    all_true_labels = np.array(all_true_labels)

    top_1_predictions = all_predicted_classes[:, 0]
    top_1_correct = top_1_predictions == all_true_labels
    top_1_accuracy = sum(top_1_correct)/len(top_1_correct)
    top_k_correct = (all_predicted_classes == all_true_labels[:, np.newaxis]).any(axis=1)
    top_k_accuracy = sum(top_k_correct)/len(top_k_correct)
    print(top_1_accuracy,top_k_accuracy)
    
    X = np.concatenate(X,axis=0)
    EEG = np.concatenate(EEG,axis=0)
    Y = np.concatenate(Y,axis=0)
    S = np.concatenate(S,axis=0)

    SX = np.concatenate(Source_X,axis=0)
    print(X.shape,SX.shape)
    
    reducer = umap.UMAP(n_neighbors=20, random_state=0)
    embedding_2d = reducer.fit_transform(EEG)
    plt.figure(figsize=(4, 4),dpi=300) 
    for label in set(S):
        indices = S == label
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],label=f'Subject {label}',s=1,alpha=1.0)
    plt.legend(loc='best', markerscale=5, fontsize='xx-small')
    os.makedirs(f'../results/{exp_dir}',exist_ok=True)
    plt.savefig(f'../results/{exp_dir}/EEG_{epoch}.png', bbox_inches='tight')  # Save the figure with tight bounding box
    
    reducer = umap.UMAP(n_neighbors=20, random_state=0)
    embedding_2d = reducer.fit_transform(X)
    plt.figure(figsize=(4, 4),dpi=300) 
    for label in set(S):
        indices = S == label
        plt.scatter(embedding_2d[indices, 0], embedding_2d[indices, 1],label=f'Subject {label}',s=1,alpha=1.0)
    plt.legend(loc='best', markerscale=5, fontsize='xx-small')

    os.makedirs(f'../results/{exp_dir}',exist_ok=True)
    plt.savefig(f'../results/{exp_dir}/Semantic_{epoch}.png', bbox_inches='tight')  # Save the figure with tight bounding box
    
    