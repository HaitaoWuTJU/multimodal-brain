import os,torch,h5py
import time
import numpy as np

import sys,random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
sys.path.append(os.getcwd())
from dataset.things_eeg import SingleEEGDataset
from torch.utils.data import DataLoader, Dataset
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def test_speed(data_dir):
    print(data_dir)
    start_time = time.time()
    for i in range(10):
        for mode in ['train','test']:
            pt_path = os.path.join(data_dir,'sub-08',f'{mode}.pt')
            data = torch.load(pt_path)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"代码运行时间: {elapsed_time:.2f} 秒")
def test_dataloader_speed(data_dir):
    selected_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2']
    model_type = 'ViT-B/32'
    test_dataset = SingleEEGDataset(data_dir, subject='sub-08', mode='test',selected_ch=selected_ch,model_type=model_type,average=True)
    train_dataset = SingleEEGDataset(data_dir, subject='sub-08', mode='train',selected_ch=selected_ch,model_type=model_type)
    
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    for epoch in range(100):
        for i,sample in enumerate(train_loader):
            eeg, label, img, img_features,text, text_features,session = sample
            eeg = eeg.to(device)
        for i,sample in enumerate(test_loader):
            eeg, label, img, img_features,text, text_features,session = sample
            eeg = eeg.to(device)
            pass
if __name__=="__main__":
    set_seed(0)
    data_dirs =[]
    data_dirs.append('/root/datasets/things-eeg-small/Preprocessed_data_250Hz') #46.33 44.73
    data_dirs.append('/dev/shm/datasets/things-eeg-small/Preprocessed_data_250Hz') #44
    data_dirs.append('/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz') #55.13
    for data_dir in data_dirs:
        # test_speed(data_dir)
        start_time = time.time()
        test_dataloader_speed(data_dir)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"代码运行时间: {elapsed_time:.2f} 秒")
        
    