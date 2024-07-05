import h5py
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from lavis.models.clip_models.loss import ClipLoss
from torch.utils.data import DataLoader, Dataset
import random
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix


import sys
sys.path.append(os.getcwd())

from models.mlp import MLP,ProjectLayer,Direct
from models.resnet import Resnet18
from models.cnn import SimpleCNN

from dataset.things import ThingsDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def set_logging(file_path):
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s',filename=file_path,force=True)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def train_subject(config,data,model,train_args,eval_args):
    for epoch in range(train_args['epochs']):
        for k,v in model.items():
            v.train()
        tot_cnt=0
        tot_correct_cnt= 0
        for i,sample in enumerate(data['train']):
    
            img, label,img_feature = sample
            img = img.to(device)
            label = label.to(device)
            img_feature = img_feature.to(device)
            
            if config['img_model']['name'] in ['Resnet18','SimpleCNN']:
                outputs = model['img'](img)
            else:
                outputs = model['img'](img_feature)
            
            loss = train_args['criterion'](outputs,label)
            train_args['optimizer'].zero_grad() 
            loss.backward()
            train_args['optimizer'].step()
    
            prediction = torch.argmax(outputs, dim=1)
            correct_count = torch.sum(prediction == label)
            tot_cnt+=prediction.shape[0]
            tot_correct_cnt+=correct_count
            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.4f}")

        logging.info(f"Epoch {epoch + 1}, Train ACC: {tot_correct_cnt/tot_cnt:.3f}")
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],'ckpt.pth'))
        
        top_1_accuracy,conf_matrix = eval_subject(config,data,model,eval_args)
        logging.info(f"Test Top1-ACC: {top_1_accuracy:.3f}")
        # logging.info(f"Test conf_matrix: {conf_matrix}")
        infos=[]
        for i in range(config['img_model']['args']['num_classes']):
            TP_i = conf_matrix[i, i]
            FN_i = conf_matrix[i, :].sum() - TP_i
            recall_i = TP_i / (TP_i + FN_i) if (TP_i + FN_i) > 0 else 0
            infos.append((f"Class {i}: {recall_i}"))
        logging.info("\t".join(infos))

@torch.no_grad()
def eval_subject(config,data,model,eval_args):
    for k,v in model.items():
        v.eval()

    all_prediction = []
    all_label = []
    for i,sample in enumerate(data['test']):
        img, label,img_feature = sample

        img = img.to(device)
        label = label.to(device)
        img_feature = img_feature.to(device)

        if config['img_model']['name'] in ['Resnet18','SimpleCNN']:
            outputs = model['img'](img)
        else:
            outputs = model['img'](img_feature)

        prediction = torch.argmax(outputs, dim=1)
        all_prediction.extend(prediction.cpu().numpy())
        all_label.extend(label.cpu().numpy())
    
    conf_matrix = confusion_matrix(all_label,all_prediction)
    
    tot_correct_cnt = conf_matrix.trace()
    tot_cnt = conf_matrix.sum()
    top_1_accuracy = tot_correct_cnt/tot_cnt
    return top_1_accuracy,conf_matrix

def main():
    set_seed(0)
    config = {
        "data_dir": "/root/workspace/wht/multimodal_brain/datasets/things",
        "exp_root":'/root/workspace/wht/multimodal_brain/src/exp',
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 16,
        "logger": True,
        "img_model": {'name':'SimpleCNN','args':{'num_classes':3}},
        # "img_model": {'name':'Resnet18',"args":{'num_classes':3, 'pretrained':False}},
        # "img_model":  {'name':'MLP',"args":{'input_dim':1024,'output_dim':4,'hiden_dims':[256]}}
    }
    
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],'exp.LOG'))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")
    
    img_model = globals()[config['img_model']['name']](**config['img_model']['args'])
    img_model.to(device)
    
    optimizer = optim.Adam(list(img_model.parameters()), lr=config['lr'])
    
    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(img_model.parameters())])}")
    
    transform = transforms.Compose([
    transforms.Resize((228, 228)),
    # transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = ThingsDataset(config['data_dir'],'train',transform)
    test_dataset = ThingsDataset(config['data_dir'], 'test',transform)
    
    print(f"train:{len(train_dataset)} test:{len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=16, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=16)
    
    data={
        'train':train_loader,
        'test':test_loader
    }
    model = {
        'img':img_model,
    }
    train_args = {
        'optimizer':optimizer,
        'criterion':nn.CrossEntropyLoss(),
        'epochs':config['epochs']
    }
    eval_args = {}
    
    train_subject(config,data,model,train_args,eval_args)

if __name__=="__main__":
    
    main()