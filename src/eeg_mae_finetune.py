import logging,torch,os,itertools,random
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset.things_eeg import EEGDataset
from models.ae import Autoencoder,Autoencoder_MLP
from torchvision import transforms
from utils import set_logging
from lavis.models.clip_models.loss import ClipLoss
from sklearn.metrics import accuracy_score
from models.mlp import MLP,ProjectLayer,Direct

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train(data,model,args):
    device = args['device']
    for epoch in range(args['epochs']):
        for k,v in model.items():
            v.train()
        tot_cnt=0+1e-8
        tot_correct_cnt= 0
        for i,sample in enumerate(data['train']):
            eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            eeg = eeg.to(device)
            label = label.to(device)
            img_features = img_features.to(device)
            
            latent = model['eeg'].forward_pool(eeg).squeeze(1)
            # latent = model['eeg'].forward_cls(eeg).squeeze(1)
            latent_proj = model['linear_project'](latent)
            
            logit_scale = model['linear_project'].logit_scale.exp()
            eeg_img_loss = args['criterion'](latent_proj, img_features,logit_scale)
            
            loss = eeg_img_loss
            
            args['optimizer'].zero_grad() 
            loss.backward()
            args['optimizer'].step()
            
            
            img_features = img_features/img_features.norm(dim=-1, keepdim=True)
            similarity = (latent_proj @ img_features.T)
            max_index = torch.argmax(similarity, dim=1)
            prediction = label[max_index]
            correct_count = torch.sum(prediction == label)
            tot_cnt+=prediction.shape[0]
            tot_correct_cnt+=correct_count
            
            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()/eeg.shape[0]:.5f}")
                
        logging.info(f"Epoch {epoch + 1}, Train ACC: {tot_correct_cnt/tot_cnt:.3f}")
        # torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(config['exp_dir'],'ckpt.pth'))
        
        top_1_accuracy,top_k_accuracy, _ = eval(data,model,args)
        
        logging.info(f"Test Top1-ACC: {top_1_accuracy:.3f},Top5-ACC: {top_k_accuracy:.3f}")
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(args['exp_dir'],'ckpt.pth'))
        
@torch.no_grad()              
def eval(data,model,args):
    device = args['device']

    for k,v in model.items():
        v.eval()
        
    n_way = args['n_way'] if 'n_way' in args.keys() else False
    all_predicted_classes = []
    all_true_labels = []
    all_predicted_classes_n_way = []
    all_true_labels_n_way = []
    
    all_image_features=data['test'].dataset.all_image_features.to(device)
    loss_list=[]
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
        eeg = eeg.to(device)
        img_features = img_features.to(device)
        
        latent = model['eeg'].forward_pool(eeg).squeeze(1)
        eeg_latent_proj =model['linear_project'](latent)
        
        logit_scale = model['linear_project'].logit_scale.exp()
        eeg_img_loss =  args['criterion'](eeg_latent_proj, img_features,logit_scale)
        loss = eeg_img_loss
        
        loss_list.append(loss.item()*eeg.shape[0])
        
        eeg_latent_proj = eeg_latent_proj/eeg_latent_proj.norm(dim=-1, keepdim=True)
        all_image_features = all_image_features/all_image_features.norm(dim=-1, keepdim=True)
        
        if n_way:
            for j in range(eeg_latent_proj.shape[0]):
                indices_exclude_j = torch.tensor([idx for idx in range(all_image_features.shape[0]) if idx != label[j]])
                indices = torch.randperm(indices_exclude_j.shape[0])[:n_way - 1].to(device)
                indices = torch.cat((indices, torch.tensor([label[j]],device=device)), dim=0)

                all_image_features_project_n_way = all_image_features[indices]
                similarity_n_way = (eeg_latent_proj[j] @ all_image_features_project_n_way.T)

                predictions_n_way  = similarity_n_way.argmax(dim=0)
                # all_predicted_classes_n_way.append(label[indices][predictions_n_way].item())
                all_predicted_classes_n_way.append(indices[predictions_n_way].item())
                all_true_labels_n_way.append(label[j].item())
        similarity = (eeg_latent_proj @ all_image_features.T)
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
    
    logging.info(f"Test Loss: {(sum(loss_list) / len(data['test'].dataset)):.5f}")
    return top_1_accuracy,top_k_accuracy,accuracy_n_way
def main():
    selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2']
    selected_ch = ['P7', 'P5', 'P3', 'P1','Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8','O1', 'Oz', 'O2']
    model_type = 'ViT-B-32'
    latend_dim_dict = {'ViT-B-16':512,'ViT-B-32':512,'ViT-L-14': 768,'RN50':1024,'RN101':512,'RN50x4':640,'ViT-H-14':1024,'ViT-g-14':1024,'ViT-bigG-14':1280}
    latend_dim = latend_dim_dict[model_type]
    config = {
        "device":torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
        "data_dir": "/dev/shm/wht/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten",
        "exp_root":'./exp',
        "name": os.path.basename(__file__).rsplit('.',1)[0],
        "lr": 1e-4,
        "epochs": 300,
        "batch_size": 2048,
        "model_type":model_type,
        "latend_dim":latend_dim,
        "logger": True,
        "subjects":['sub-08'],#'sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
        # "eeg_model":{'name':'Autoencoder','args':{'patch_size':25,'mlp_ratio':1.0}},
        "eeg_model":{'name':'Autoencoder_MLP','args':{}},
        "linear_project":{'name':'ProjectLayer','args':{'embedding_dim': latend_dim, 'proj_dim':latend_dim}},#{'name':"MLP","args":{'input_dim':512,'output_dim':512,'hiden_dims':[]}},
    }
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],f"{config['subjects']}.LOG"))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")

    logging.info(f"Start training on {config['subjects']}")
    
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
        'test':test_loader,
    }
    model = {
        'eeg':globals()[config['eeg_model']['name']](**config['eeg_model']['args']).to(config['device']),
        'linear_project': globals()[config['linear_project']['name']](**config['linear_project']['args']).to(config['device']),
    }

    model['eeg'].load_state_dict(torch.load(os.path.join(config['exp_root'],'eeg_mae_mlp','ckpt_sub-01_sub-02_sub-03_sub-04_sub-05_sub-06_sub-07_sub-08_sub-09_sub-10.pth'),map_location=config['device'])['eeg'])
    
    for param in model['eeg'].parameters():
        param.requires_grad = False
        
    train_eval_args = {
        'optimizer':optim.Adam([{'params': v.parameters(), 'lr': config['lr']} for k,v in model.items()]),
        'criterion':ClipLoss(),
        'epochs':config['epochs'],
        'device':config['device'],
        'exp_dir':config['exp_dir'],
    }
    train(data,model,train_eval_args)

    accuracy_n_way={'10':[],'4':[],'2':[]}
    for i in range(3):
        for n_way in [10,4,2]:
            top_1_accuracy,top_k_accuracy,_ = eval(config,data,model,{'n_way': n_way })
            accuracy_n_way[str(n_way)].append(_)
    for k,v in accuracy_n_way.items():
        logging.info(f"{k}-way Acc: {np.mean(v):.3f}")
if __name__=="__main__":
    set_seed(0)
    main()
    
   