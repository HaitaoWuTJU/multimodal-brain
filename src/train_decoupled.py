import h5py
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

from lavis.models.clip_models.loss import ClipLoss
from torch.utils.data import DataLoader, Dataset
import random
import itertools
from sklearn.metrics import accuracy_score
import argparse
import ast

from models.mlp import MLP,ProjectLayer,Direct
from models.eeg import EEGEncoder,LSTMModel

from dataset.things_eeg import EEGDataset


class ZScoreNormalize:
    def __call__(self, sample):
        mean = sample.mean(dim=1, keepdim=True)
        std = sample.std(dim=1, keepdim=True)
        normalized_sample = (sample - mean) / (std + 1e-8)
        return normalized_sample
    
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
    criterion = nn.CrossEntropyLoss()
    for epoch in range(train_args['epochs']):

        for k,v in model.items():
            v.train()
        tot_cnt=0+1e-8
        tot_correct_cnt= 0
        for i,sample in enumerate(data['train']):
            
            eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            eeg = eeg.to(device) #.float()
            label = label.to(device)
            subject = subject.to(device)

            img_features = img_features.to(device)
            text_features = text_features.to(device)
            
            eeg_features = model['eeg'](eeg)
            img_features_project =  model['img'](img_features)
            text_features_project =  model['text'](text_features)

            logit_scale = model['eeg'].logit_scale.exp()
            
            eeg_img_loss =  train_args['criterion'](eeg_features, img_features_project,logit_scale)
            eeg_text_loss =  train_args['criterion'](eeg_features, text_features_project,logit_scale)
            text_img_loss = train_args['criterion'](text_features,img_features,logit_scale)
            
            eeg_logits = model['linear_probe'](eeg_features)
            prediction = torch.argmax(eeg_logits, dim=1)
            cls_loss = criterion(eeg_logits,subject)
            
            loss = cls_loss
            train_args['optimizer'].zero_grad()
            loss.backward()
            train_args['optimizer'].step()
            
            correct_count = torch.sum(prediction == subject)
            tot_cnt+=prediction.shape[0]
            tot_correct_cnt+=correct_count
            
            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.4f}")
                
        logging.info(f"Epoch {epoch + 1}, Train ACC: {tot_correct_cnt/tot_cnt:.3f}")
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],f"ckpt_{'_'.join(train_args['subjects'])}.pth"))
        
        top_1_accuracy = eval_subject(config,data,model,eval_args)
        logging.info(f"Test Top1-ACC: {top_1_accuracy:.4f}")

@torch.no_grad()
def eval_subject(config,data,model,eval_args):
    for k,v in model.items():
        v.eval()
    all_predicted = []
    tot_cnt=0
    tot_correct_cnt=0
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features,session,subject   = sample

        eeg = eeg.to(device)
        subject = subject.to(device)
        eeg_features = model['eeg'](eeg)
        eeg_logits = model['linear_probe'](eeg_features)

        prediction = torch.argmax(eeg_logits, dim=1)
        tot_cnt+=subject.shape[0]
        tot_correct_cnt+=sum(prediction==subject)
    top_1_accuracy = tot_correct_cnt/tot_cnt
    # logging.info(f"Test Top1-ACC: {top_1_accuracy:.4f}")
    return top_1_accuracy

def get_args_parser():
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--subjects', type=str)

    return parser

def update_config(args, config):
    for key in config.keys():
        if hasattr(args, key):
            if getattr(args, key) != None:
                config[key] = getattr(args, key)
    return config

def main():
    args = get_args_parser()
    args = args.parse_args()
    
    if args.subjects:
        args.subjects= eval(args.subjects)
    
    set_seed(0)
    selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
    selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2']
    latend_dim = 512
    config = {
        "data_dir": "/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten",#"/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz/",
        "exp_root":'./exp',
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 1024,
        "model_type":'ViT-B-32', #ViT-bigG-14,ViT-H-14
        "logger": True,
        "subjects":['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
        "eeg_model":{'name':'MLP','args':{'input_dim':len(selected_ch)*250,'output_dim':latend_dim,'hiden_dims':[]}},#{'name':'LSTMModel', 'args':{}},
        "linear_probe":{'name':"MLP","args":{'input_dim':512,'output_dim':10,'hiden_dims':[]}},
        "img_model": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
        "text_model": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
    }
    config = update_config(args, config)
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    config['device'] = device
    
    data_dir = config['data_dir']
    subjects = config['subjects']
    
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],f"{'_'.join(subjects)}.LOG"))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")

    
    eeg_model = globals()[config['eeg_model']['name']](**config['eeg_model']['args'])
    eeg_model.to(device)
    logging.info(f'eeg_model:{eeg_model}')
    
    img_model = globals()[config['img_model']['name']](**config['img_model']['args'])
    img_model.to(device)
    
    text_model = globals()[config['text_model']['name']](**config['text_model']['args'])
    text_model.to(device)
    
    linear_probe_model = globals()[config['linear_probe']['name']](**config['linear_probe']['args'])
    linear_probe_model.to(device)

    
    
    transform = transforms.Compose([
    #    ZScoreNormalize()
    ])

    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(eeg_model.parameters())])}")
    test_dataset = EEGDataset(data_dir=config['data_dir'],subjects=subjects,model_type=config['model_type'],mode='test',transform=transform,avg=False)
    train_dataset = EEGDataset(data_dir=config['data_dir'],subjects=subjects,model_type=config['model_type'],mode='train',transform=transform,avg=False)
    
    logging.info(f"train num: {len(train_dataset)}, test num: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    data={
        'train':train_loader,
        'test':test_loader
    }
    model = {
        'eeg':eeg_model,
        'img':img_model,
        'text':text_model,
        'linear_probe':linear_probe_model,
    }

    train_args = {
        'optimizer': optim.Adam([{'params': v.parameters(), 'lr': config['lr']} for k,v in model.items()]),
        'criterion':ClipLoss(),
        'epochs':config['epochs'],
        'subjects':config['subjects']
    }
    eval_args = {
    }
    
    logging.info(f"Start training on {subjects}")
    train_subject(config,data,model,train_args,eval_args)
if __name__=="__main__":
    
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Time: {end_time - start_time} seconds.")