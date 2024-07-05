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
from models.ae import Autoencoder

from dataset.things_eeg import EEGDataset


class ZScoreNormalize:
    def __call__(self, sample):
        mean = sample.mean(dim=1, keepdim=True)
        std = sample.std(dim=1, keepdim=True)
        normalized_sample = (sample - mean) / (std + 1e-8)
        return normalized_sample
    
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

def train_subject(config,data,model,args):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(args['epochs']):

        for k,v in model.items():
            v.train()
        tot_cnt=0+1e-8
        tot_correct_cnt= 0
        
        tot_cnt_subject=0+1e-8
        tot_correct_cnt_subject=0
        for i,sample in enumerate(data['train']):
            
            eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            eeg = eeg.to(device)
            label = label.to(device)
            session = session.long().to(device)
            subject = subject.long().to(device)
            

            img_features = img_features.to(device)
            text_features = text_features.to(device)
            
            # eeg_features = model['eeg'](eeg)
            eeg_features = model['eeg'].forward_cls(eeg)
            # print(eeg_features.shape)
            semantic_features = eeg_features[:,:512]
            bias_features = eeg_features[:,-512:]
            
            semantic_logits = model['aux'](semantic_features)
            bias_logits = model['aux2'](bias_features)
            target_tensor = torch.full((eeg_features.shape[0], 10), 0.25, device=device)
            prediction = torch.argmax(semantic_logits, dim=1)
            tot_cnt_subject+=eeg.shape[0]
            tot_correct_cnt_subject+=sum(prediction==subject)
    
            semantic_cls_loss = criterion(semantic_logits,target_tensor)
            # bias_cls_loss = criterion(bias_logits,subject)
            
            logit_scale = model['eeg'].logit_scale#.exp()
            
            eeg_img_loss =  args['criterion'](semantic_features, img_features, logit_scale)
            eeg_text_loss =  args['criterion'](semantic_features, text_features,logit_scale)
            text_img_loss = args['criterion'](text_features,img_features,logit_scale)
            
            loss = 1.0*eeg_img_loss #+ 0.2*semantic_cls_loss #+ 0.2*bias_cls_loss
            args['optimizer'].zero_grad()
            loss.backward()
            args['optimizer'].step()
            semantic_features = semantic_features/semantic_features.norm(dim=-1, keepdim=True)
            img_features = img_features/img_features.norm(dim=-1, keepdim=True)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            
            similarity = (semantic_features @ img_features.T)
            max_index = torch.argmax(similarity, dim=1)
            prediction = label[max_index]
            correct_count = torch.sum(prediction == label)
            tot_cnt+=prediction.shape[0]
            tot_correct_cnt+=correct_count
            
            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item():.4f} text_img_loss:{text_img_loss.item():.4f}")
         
              
        train_acc =  tot_correct_cnt/tot_cnt
        logging.info(f"Epoch {epoch + 1}, Train ACC: {train_acc:.3f}")
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],f"ckpt_{'_'.join(args['subjects'])}.pth"))
        
        logging.info(f"Train Subject-ACC: {tot_correct_cnt_subject/tot_cnt_subject:.3f}")
        top_1_accuracy,top_k_accuracy,accuracy_n_way = eval_subject(config,data,model,args)
        
        tag = ''
        if top_1_accuracy > args['best_acc']:
            tag +=' best'
            args['best_acc'] = top_1_accuracy
            args['best_epoch'] = epoch
            args['best_train_acc'] = train_acc
    
            torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],f"best_ckpt_{'_'.join(args['subjects'])}.pth"))
        logging.info(f"Test Top1-ACC: {top_1_accuracy:.3f},Top5-ACC: {top_k_accuracy:.3f} {tag}")
        
    return args
@torch.no_grad()
def eval_subject(config,data,model,args):
    
    for k,v in model.items():
        v.eval()
    n_way = args['n_way'] if 'n_way' in args.keys() else False
    all_predicted_classes = []
    all_true_labels = []
    all_predicted_classes_n_way = []
    all_true_labels_n_way = []
    
    all_text_features=data['test'].dataset.all_text_features
    all_image_features=data['test'].dataset.all_image_features

    all_text_features = all_text_features/all_text_features.norm(dim=-1, keepdim=True)
    all_image_features = all_image_features/all_image_features.norm(dim=-1, keepdim=True)
    all_image_features = all_image_features.to(device)
    
    tot_cnt_subject=0+1e-8
    tot_correct_cnt_subject=0
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features ,session,subject = sample

        eeg = eeg.to(device) #
        label = label.to(device)
        text_features = text_features.to(device)
        img_features = img_features.to(device)
        session = session.long().to(device)
        subject = subject.long().to(device)
        
        # eeg_features = model['eeg'](eeg)
        eeg_features = model['eeg'].forward_cls(eeg)
        
        semantic_features = eeg_features[:,:512]
        bias_features = eeg_features[:,-512:]
        
        semantic_logits = model['aux'](semantic_features)
        bias_logits = model['aux2'](bias_features)
        target_tensor = torch.full((eeg_features.shape[0], 10), 0.25, device=device)
        prediction = torch.argmax(semantic_logits, dim=1)
        tot_cnt_subject+=eeg.shape[0]
        tot_correct_cnt_subject+=sum(prediction==subject)
        
        semantic_features = semantic_features/semantic_features.norm(dim=-1, keepdim=True)
        img_features = img_features/img_features.norm(dim=-1, keepdim=True)
        
        if n_way:
            for j in range(img_features.shape[0]):
                indices_exclude_j = torch.tensor([idx for idx in range(all_image_features.shape[0]) if idx != label[j]])
                indices = torch.randperm(indices_exclude_j.shape[0])[:n_way - 1].to(device)
                indices = torch.cat((indices, torch.tensor([label[j]],device=device)), dim=0)

                all_image_features_n_way = all_image_features[indices]
                # all_text_features_project_n_way = all_text_features_project[indices]

                similarity_n_way = (semantic_features[j] @ all_image_features_n_way.T)
                # similarity_n_way = (eeg_features[j] @ all_text_features_project_n_way.T)

                predictions_n_way  = similarity_n_way.argmax(dim=0)
                # all_predicted_classes_n_way.append(label[indices][predictions_n_way].item())
                all_predicted_classes_n_way.append(indices[predictions_n_way].item())
                all_true_labels_n_way.append(label[j].item())

        similarity = (semantic_features @ all_image_features.T)
        top_kvalues, top_k_indices = similarity.topk(5, dim=-1)
        all_predicted_classes.append(top_k_indices.cpu().numpy())
        all_true_labels.extend(label.cpu().numpy())

    # logging.info(f"Test Subject-ACC: {tot_correct_cnt_subject/tot_cnt_subject:.3f}")
    
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
    parser.add_argument('--subjects', type=str)
    parser.add_argument('--name', type=str)

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
    # selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
    #                                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
    #                                         'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
    #                                         'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
    #                                         'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
    #                                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    #                                         'O1', 'Oz', 'O2']
    model_type = 'ViT-B-32'
    latend_dim_dict = {'ViT-B-16':512,'ViT-B-32':512,'ViT-L-14': 768,'RN50':1024,'RN101':512,'RN50x4':640,'ViT-H-14':1024,'ViT-g-14':1024,'ViT-bigG-14':1280}
    latend_dim = latend_dim_dict[model_type]
    config = {
        "data_dir": "/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten",#"/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz/",
        "exp_root":'./exp',
        "device":device,
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 128,
        "model_type":model_type,
        "latend_dim":latend_dim,
        "logger": True,
        "subjects": ['sub-08'],#['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
        # "eeg":{'name':'ProjectLayer','args':{'embedding_dim':len(selected_ch)*250, 'proj_dim':latend_dim}},
        "eeg":{'name':'Autoencoder','args':{'in_chans':len(selected_ch)}},
        # "eeg_model":{'name':'MLP','args':{'input_dim':len(selected_ch)*250,'output_dim':latend_dim,'hiden_dims':[]}},#{'name':'LSTMModel', 'args':{}},
        "img": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
        "text": {'name':'Direct','args':{}},  # {'name':'ProjectLayer', 'args':{'embedding_dim':1024, 'proj_dim':1024}},
        "aux":{'name':'MLP','args':{'input_dim':latend_dim_dict[model_type],'output_dim':10,'hiden_dims':[]}},
        "aux2":{'name':'MLP','args':{'input_dim':latend_dim_dict[model_type],'output_dim':10,'hiden_dims':[]}},
    }
    config = update_config(args, config)
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    
    
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],f"{'_'.join(config['subjects'])}.LOG"))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")


    transform = transforms.Compose([
    #    ZScoreNormalize()
    ])

    
    test_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='test',selected_ch=selected_ch,transform=transform,avg=True)
    train_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='train',selected_ch=selected_ch,transform=transform,avg=False)
    
    logging.info(f"train num: {len(train_dataset)}, test num: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    data={
        'train':train_loader,
        'test':test_loader
    }
    model = {
        'eeg':globals()[config['eeg']['name']](**config['eeg']['args']).to(device),
        'img':globals()[config['img']['name']](**config['img']['args']).to(device),
        'text':globals()[config['text']['name']](**config['text']['args']).to(device),
        'aux':globals()[config['aux']['name']](**config['aux']['args']).to(device),
        'aux2':globals()[config['aux2']['name']](**config['aux2']['args']).to(device),
    }
    logging.info(f"eeg_model:{model['eeg']}")
    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(model['eeg'].parameters())])}")
    
    train_eval_args = {
        'optimizer': optim.Adam([{'params': v.parameters(), 'lr': config['lr']} for k,v in model.items()]),
        'criterion':ClipLoss(),
        'epochs':config['epochs'],
        'subjects':config['subjects'],
        'best_acc':-1,
        'best_epoch':-1,
        'best_train_acc':-1,
    }
    
    logging.info(f"Start training on {config['subjects']}")
    train_eval_args = train_subject(config,data,model,train_eval_args)
    
    
    logging.info(f"Start eval on {config['subjects']}")
    model['eeg'].load_state_dict(torch.load(os.path.join(config['exp_dir'],f"best_ckpt_{'_'.join(config['subjects'])}.pth"),map_location=config['device'])['eeg'])
    
    accuracy_n_way={'10':[],'4':[],'2':[]}
    for i in range(3):
        for n_way in [10,4,2]:
            top_1_accuracy,top_k_accuracy,_ = eval_subject(config,data,model,{'n_way': n_way })
            accuracy_n_way[str(n_way)].append(_)
            
    logging.info(f"Train ACC: {train_eval_args['best_train_acc']:.3f}")
    
    logging.info(f"Best@Epoch-{train_eval_args['best_epoch']}, Test Top1-ACC: {top_1_accuracy:.3f},Top5-ACC: {top_k_accuracy:.3f}")
    
    for k,v in accuracy_n_way.items():
        logging.info(f"{k}-way Acc: {np.mean(v):.3f}")
        
if __name__=="__main__":
    
    start_time = time.time()
    main()
    end_time = time.time()
    logging.info(f"Time: {end_time - start_time} seconds.")