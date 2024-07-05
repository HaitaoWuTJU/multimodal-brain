import logging,torch,os,itertools,random
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from dataset.things_eeg import EEGDataset
from models.ae import Autoencoder,Autoencoder_MLP
from torchvision import transforms
from utils import set_logging
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
    criterion = args['criterion']
    for epoch in range(args['epochs']):
        for k,v in model.items():
            v.train()
            
        for i,sample in enumerate(data['train']):
            eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
            eeg = eeg.to(device)
            
            loss, pred, mask= model['eeg'](eeg)
            args['optimizer'].zero_grad() 
            loss.backward()
            args['optimizer'].step()
            
            if (i+1) % 10 == 0:
                logging.info(f"Epoch {epoch + 1}, Iteration {i}, Loss: {loss.item()/eeg.shape[0]:.5f}")
        eval(data,model,args)
        torch.save({k:v.state_dict() for k,v in model.items()}, os.path.join(args['exp_dir'],f"ckpt_{'_'.join(args['subjects'])}.pth"))
        
@torch.no_grad()              
def eval(data,model,args):
    device = args['device']
    criterion = args['criterion']

    for k,v in model.items():
        v.eval()
            
    loss_list=[]
    for i,sample in enumerate(data['test']):
        eeg, label, img, img_features,text, text_features,session,subject = sample #x:[63, 250], label:1024 text:text text_features:[1024, 1024] img:1024 img_features:[1024, 1024]
        eeg = eeg.to(device)
        
        loss, pred, mask= model['eeg'](eeg)
        loss_list.append(loss.item()*eeg.shape[0])
        
    logging.info(f"Test Loss: {(sum(loss_list) / len(data['test'].dataset)):.5f}")
    
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
        "name": os.path.basename(__file__).rsplit('.',1)[0]+'_mlp',
        "lr": 1e-4,
        "epochs": 300,
        "batch_size": 2048,
        "model_type":model_type,
        "latend_dim":latend_dim,
        "logger": True,
        "subjects":['sub-01','sub-02','sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10'],
        "eeg_model":{'name':'Autoencoder_MLP','args':{}}#'Autoencoder','args':{'patch_size':25,'mlp_ratio':1.0}},
    }
    config['exp_dir'] = os.path.join(config['exp_root'],config['name'])
    os.makedirs(config['exp_dir'],exist_ok=True)
    set_logging(os.path.join(config['exp_dir'],f"{config['subjects']}.LOG"))
    logging.info(f"-------------------------START-------------------------")
    logging.info(f"CONFIG: {config}")

    logging.info(f"Start training on {config['subjects']}")
    
    test_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='test')
    train_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='train')
    
    
    model = {
        'eeg':globals()[config['eeg_model']['name']](**config['eeg_model']['args']).to(config['device']),
    }
    
    
    transform = transforms.Compose([
    #    ZScoreNormalize()
    ])
    logging.info(f"Number of parameters: {sum([p.numel() for p in itertools.chain(model['eeg'].parameters())])}")
    test_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='test',selected_ch=selected_ch,transform=transform,avg=True)
    train_dataset = EEGDataset(data_dir=config['data_dir'],subjects=config['subjects'],model_type=config['model_type'],mode='train',selected_ch=selected_ch,transform=transform,avg=False)
    
    logging.info(f"train num: {len(train_dataset)}, test num: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    data={
        'train':train_loader,
        'test':test_loader,
    }
    

    train_eval_args = {
        'optimizer':optim.Adam(model['eeg'].parameters(), lr=config['lr']),
        'criterion':nn.MSELoss(),
        'epochs':config['epochs'],
        'device':config['device'],
        'exp_dir':config['exp_dir'],
        'subjects':config['subjects']
    }
    train(data,model,train_eval_args)

if __name__=="__main__":
    set_seed(0)
    main()
    
   