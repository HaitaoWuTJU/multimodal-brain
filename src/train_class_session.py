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
from sklearn.metrics import accuracy_score


from models.mlp import MLP,ProjectLayer,Direct


from dataset.datasets import SingleEEGDataset


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
            
            eeg, label, img, img_features,text, text_features, session = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            # logging.info(f"{x.shape} {label.shape} {len(text)} {text_features.shape} {len(img)} {img_features.shape}")
            eeg = eeg.to(device)
            session = session.to(device)

            eeg_features = model['eeg'](eeg)
            eeg_logits = model['linear_probe'](eeg_features)

            prediction = torch.argmax(eeg_logits, dim=1)
            
            loss = criterion(eeg_logits, session)
            train_args['optimizer'].zero_grad()
            loss.backward()
            train_args['optimizer'].step()
            tot_cnt+=session.shape[0]
            tot_correct_cnt+=sum(prediction==session)

            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.5f}")
        logging.info(f"Epoch {epoch + 1}, Train ACC: {tot_correct_cnt/tot_cnt:.4f}")

        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],'ckpt.pth'))
        eval_subject(config,data,model,eval_args)

@torch.no_grad()
def eval_subject(config,data,model,eval_args):
    for k,v in model.items():
        v.eval()
    all_predicted = []
    tot_cnt=0
    tot_correct_cnt=0
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features,session  = sample

        eeg = eeg.to(device)
        session = session.to(device)
        eeg_features = model['eeg'](eeg)
        eeg_logits = model['linear_probe'](eeg_features)

        prediction = torch.argmax(eeg_logits, dim=1)
        tot_cnt+=session.shape[0]
        tot_correct_cnt+=sum(prediction==session)
    
    
    logging.info(f"Test Top1-ACC: {tot_correct_cnt/tot_cnt:.4f}")

def main():
    
    selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
    selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2']
    config = {
        "data_dir": "/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz",
        "exp_root":'./exp',
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 1024,
        "logger": True,
        "insubject": True,
        "subject":'sub-10', # ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
        "selected_ch": selected_ch,
        "eeg_model": {'name':'MLP',"args":{'input_dim':len(selected_ch)*250,'output_dim':1024,'hiden_dims':[]}},
        "img_model": {'name':'Direct',"args":{}},  # {'name':'ProjectLayer', "args":{'embedding_dim':1024, 'proj_dim':1024}},
        "text_model": {'name':'Direct',"args":{}},  # {'name':'ProjectLayer', "args":{'embedding_dim':1024, 'proj_dim':1024}},
        "linear_probe":{'name':"MLP","args":{'input_dim':1024,'output_dim':4,'hiden_dims':[]}},
    }

    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    
    
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],'exp.LOG'))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")

    data_dir = config['data_dir']
    subject = config['subject']
    logging.info(f"Start training on {subject}")
    eeg_model = globals()[config['eeg_model']['name']](**config['eeg_model']['args'])
    ckpt = torch.load('/root/workspace/wht/multimodal_brain/src/exp/train_align_CLIP/ckpt.pth')
    eeg_model.load_state_dict(ckpt['eeg'])
    eeg_model.to(device)
    
    for param in eeg_model.parameters():
        param.requires_grad = False

    img_model = globals()[config['img_model']['name']](**config['img_model']['args'])
    img_model.to(device)
    
    text_model = globals()[config['text_model']['name']](**config['text_model']['args'])
    text_model.to(device)
    
    linear_probe_model = globals()[config['linear_probe']['name']](**config['linear_probe']['args'])
    linear_probe_model.to(device)

    optimizer = optim.Adam(list(eeg_model.parameters()) + list(img_model.parameters()), lr=config['lr'])

    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(eeg_model.parameters())])}")
    
    train_dataset = SingleEEGDataset(data_dir, subject=subject, mode='train',selected_ch=config['selected_ch'])
    test_dataset = SingleEEGDataset(data_dir, subject=subject, mode='test',selected_ch=config['selected_ch'])
    logging.info(f"train num: {len(train_dataset)}, test num: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    
    
    data={
        'train':train_loader,
        'test':test_loader
    }
    model = {
        'eeg':eeg_model,
        'img':img_model,
        'text':text_model,
        'linear_probe':linear_probe_model
    }
    train_args = {
        'optimizer':optimizer,
        'eeg_criterion':ClipLoss(),
        'epochs':config['epochs']
    }
    eval_args = {
    }
    train_subject(config,data,model,train_args,eval_args)
    logging.info(f"Start eval on {subject}")
    eval_subject(config,data,model,eval_args)
        
        
if __name__=="__main__":
    set_seed(0)
    main()