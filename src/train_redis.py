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
import argparse

from models.mlp import MLP,ProjectLayer,Direct
from models.eeg import EEGEncoder

from dataset.things_eeg import SingleEEGDataset,EEGDatasetRedis

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
        tot_cnt=0+1e-8
        tot_correct_cnt= 0
        for i,sample in enumerate(data['train']):
            
            eeg, label, img, img_features,text, text_features,session = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            eeg = eeg.float().to(device)
            label = label.to(device)


            img_features = img_features.to(device)
            text_features = text_features.to(device)
            
            
            eeg_features = model['eeg'](eeg)
            img_features_project =  model['img'](img_features)
            text_features_project =  model['text'](text_features)

            logit_scale = model['eeg'].logit_scale.exp()
            
            eeg_img_loss =  train_args['eeg_criterion'](eeg_features, img_features_project,logit_scale)
            eeg_text_loss =  train_args['eeg_criterion'](eeg_features, text_features_project,logit_scale)
            text_img_loss = train_args['eeg_criterion'](text_features,img_features,logit_scale)
            
            loss = eeg_img_loss
            train_args['optimizer'].zero_grad() 
            loss.backward()
            train_args['optimizer'].step()
            eeg_features = eeg_features/eeg_features.norm(dim=-1, keepdim=True)
            
            img_features_project = img_features_project/img_features_project.norm(dim=-1, keepdim=True)
            text_features_project = text_features_project/text_features_project.norm(dim=-1, keepdim=True)
            
            similarity = (eeg_features @ img_features_project.T)
            max_index = torch.argmax(similarity, dim=1)
            prediction = label[max_index]
            correct_count = torch.sum(prediction == label)
            tot_cnt+=prediction.shape[0]
            tot_correct_cnt+=correct_count
            

            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.4f} text_img_loss:{text_img_loss.item():.4f}")
                
        logging.info(f"Epoch {epoch + 1}, Train ACC: {tot_correct_cnt/tot_cnt:.3f}")
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],'ckpt.pth'))
        
        # top_1_accuracy,top_k_accuracy,accuracy_n_way = eval_subject(config,data,model,eval_args)
        # logging.info(f"Test Top1-ACC: {top_1_accuracy:.3f},Top5-ACC: {top_k_accuracy:.3f}")

@torch.no_grad()
def eval_subject(config,data,model,eval_args):
    
    for k,v in model.items():
        v.eval()
    n_way = eval_args['n_way'] if 'n_way' in eval_args.keys() else False
    all_predicted_classes = []
    all_true_labels = []
    all_predicted_classes_n_way = []
    all_true_labels_n_way = []
    
    # all_text_features=data['test'].dataset.all_text_features
    all_image_features=data['test'].dataset.all_image_features

    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features ,session = sample

        eeg = eeg.float().to(device)
        label = label.to(device)

        text_features = text_features.to(device)
        img_features = img_features.to(device)
        all_image_features = all_image_features.to(device)

        eeg_features = model['eeg'](eeg)
        
        img_features_project =  model['img'](img_features)
        all_image_features_project =  model['img'](all_image_features)
        
        # all_text_features_project =  model['text'](all_text_features)
        # text_features_project = model['text'](text_features)

        eeg_features = eeg_features/eeg_features.norm(dim=-1, keepdim=True)
        img_features_project = img_features_project/img_features_project.norm(dim=-1, keepdim=True)
        # text_features_project = text_features_project/text_features_project.norm(dim=-1, keepdim=True)
        all_image_features_project = all_image_features_project/all_image_features_project.norm(dim=-1, keepdim=True)
        
        if n_way:
            for j in range(img_features_project.shape[0]):
                indices_exclude_j = torch.tensor([idx for idx in range(all_image_features_project.shape[0]) if idx != label[j]])
                indices = torch.randperm(indices_exclude_j.shape[0])[:n_way - 1].to(device)
                indices = torch.cat((indices, torch.tensor([label[j]],device=device)), dim=0)

                all_image_features_project_n_way = all_image_features_project[indices]
                # text_features_project_n_way = text_features_project[indices]

                similarity_n_way = (eeg_features[j] @ all_image_features_project_n_way.T)

                predictions_n_way  = similarity_n_way.argmax(dim=0)
                # all_predicted_classes_n_way.append(label[indices][predictions_n_way].item())
                all_predicted_classes_n_way.append(indices[predictions_n_way].item())
                all_true_labels_n_way.append(label[j].item())
            
        similarity = (eeg_features @ all_image_features_project.T)
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
    
    if n_way:
        accuracy_n_way = accuracy_score(all_true_labels_n_way, all_predicted_classes_n_way)
    else:
        accuracy_n_way = None
    return top_1_accuracy,top_k_accuracy,accuracy_n_way


def get_args_parser():
    parser = argparse.ArgumentParser('train', add_help=False)
    # project parameters
    parser.add_argument('--subject', type=str)
    
    # finetune parameters
   

    # diffusion sampling parameters
   
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
        "data_root": "/root/workspace/wht/multimodal_brain/datasets/things-eeg-small",
        "exp_root":'./exp',
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 1024,
        "logger": True,
        "insubject": True,
        "subject":'sub-08', # ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
        "selected_ch": selected_ch,
        "eeg_model":{'name':'MLP','args':{'input_dim':len(selected_ch)*250,'output_dim':latend_dim,'hiden_dims':[]}},
        "img_model": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
        "text_model": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
    }
    config = update_config(args, config)
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    
    data_root = config['data_root']
    subject = config['subject']
    
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],f'{subject}.LOG'))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")

    logging.info(f"Start training on {subject}")
    eeg_model = globals()[config['eeg_model']['name']](**config['eeg_model']['args'])
    eeg_model.to(device)

    img_model = globals()[config['img_model']['name']](**config['img_model']['args'])
    img_model.to(device)
    
    text_model = globals()[config['text_model']['name']](**config['text_model']['args'])
    text_model.to(device)

    optimizer = optim.Adam(list(eeg_model.parameters()) + list(img_model.parameters()), lr=config['lr'])

    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(eeg_model.parameters())])}")
    
    train_dataset = EEGDatasetRedis(data_root,subjects=['sub-10'],mode='train',selected_ch=config['selected_ch'],model_type='ViT-B/32')
    
    logging.info(f"train num: {len(train_dataset)}")#, test num: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    
    data={
        'train':train_loader,
        # 'test':test_loader
    }
    model = {
        'eeg':eeg_model,
        'img':img_model,
        'text':text_model
    }
    train_args = {
        'optimizer':optimizer,
        'eeg_criterion':ClipLoss(),
        'epochs':config['epochs']
    }
    eval_args = {
    }
    train_subject(config,data,model,train_args,eval_args)
    
    # logging.info(f"Start eval on {subject}")
    # accuracy_n_way={'10':[],'4':[],'2':[]}
    # for i in range(5):
    #     for n_way in [10,4,2]:
    #         top_1_accuracy,top_k_accuracy,_ = eval_subject(config,data,model,{'n_way': n_way })
    #         accuracy_n_way[str(n_way)].append(_)
    # for k,v in accuracy_n_way.items():
    #     logging.info(f"{k}-way Acc: {np.mean(v):.3f}")
        
if __name__=="__main__":
    
    main()