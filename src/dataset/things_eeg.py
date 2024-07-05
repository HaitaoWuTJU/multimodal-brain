import torch,os,mne
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import redis
import pickle

cuda_device_count = torch.cuda.device_count()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, subject_id=1,data_root='/root/workspace/wht/multimodal_brain/datasets/things-eeg'):
        self.subject_id = subject_id
        self.eeg_data_dir = os.path.join(data_root,f'sub-{subject_id:02}','eeg')

        self.eeg_data_path = os.path.join(self.eeg_data_dir,f'sub-{subject_id:02}_task-rsvp_eeg.eeg')
        self.eeg_vhdr_path = os.path.join(self.eeg_data_dir,f'sub-{subject_id:02}_task-rsvp_eeg.vhdr')
        self.eeg_vmrk_path = os.path.join(self.eeg_data_dir,f'sub-{subject_id:02}_task-rsvp_eeg.vmrk')
        self.eeg_json_path = os.path.join(self.eeg_data_dir,f'sub-{subject_id:02}_task-rsvp_eeg.json')
        self.eeg_event_path = os.path.join(self.eeg_data_dir,f'sub-{subject_id:02}_task-rsvp_events.tsv')
        
        raw = mne.io.read_raw_brainvision(self.eeg_vhdr_path, preload=True)
        self.df = pd.read_csv(self.eeg_event_path, delimiter='\t')
        
        # montage = mne.channels.read_custom_montage('BC-22-re.bvef')
        # raw.set_montage(montage)
        # print(raw.info)

        events = mne.events_from_annotations(raw)
        
        
        # raw.pick_channels(chan_order, ordered=True) 
        print(events)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample


class EEGDataset():
    def __init__(self, data_dir, subjects, mode,model_type = 'ViT-H-14',selected_ch=None,transform=None,avg=False):
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
        
        self.transform = transform
        self.avg = avg
        self.n_cls = 1654 if self.mode=='train' else 200
        self.data_paths = [os.path.join(self.data_dir,subject,f'{mode}.pt') for subject in self.subjects]
        
        self.loaded_data= [self.load_data(data_path) for data_path in self.data_paths]
        
        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject*len(self.subjects)
        
        
        
        features_filename = os.path.join(self.data_dir,f"{model_type.replace('/','-')}_features_{mode}.pt")
        if os.path.exists(features_filename) :
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        else:
            pretrain_map={
                
                'RN50':'openai',
                'RN101':'openai',
                'RN50x4':'openai',
                'ViT-B-16':'laion2b_s34b_b88k',
                'ViT-B-32':'laion2b_s34b_b79k',
                'ViT-L-14':'laion2b_s32b_b82k',
                'ViT-H-14':'laion2b_s32b_b79k',
                'ViT-g-14':'laion2b_s34b_b88k', 
                'ViT-bigG-14':'laion2b_s39b_b160k',
            }
            self.vlmodel, self.preprocess,_ = open_clip.create_model_and_transforms(model_type, device="cuda",pretrained=pretrain_map[model_type]) #laion2b_s39b_b160k
            self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
            }, features_filename)
        
        self.all_text_features = torch.from_numpy(np.concatenate([self.text_features[k].unsqueeze(0) for k in self.all_class_names])).to(device)
        if mode =='test':
            self.all_image_features = torch.from_numpy(np.concatenate([self.img_features[k].unsqueeze(0) for k in self.img_features])).to(device)
    
    def load_data(self,data_path):
        print(f"----load {data_path.rsplit('250HZ',1)[-1]}----")
        loaded_data = torch.load(data_path)
        loaded_data['eeg']=torch.from_numpy(loaded_data['eeg'])
        
        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:,:,selected_idx]
        if self.avg:
            avg_data={}
            label = loaded_data['label']
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
        for i in range(0, len(set_images), batch_size):
            batch_images = set_images[i:i + batch_size]
            image_inputs = torch.stack([self.preprocess(Image.open(os.path.join(self.data_dir,'../Image_set',img)).convert("RGB")) for img in batch_images]).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]:image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict
    
    @torch.no_grad()
    def Textencoder(self, text):   
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"This is a {t}.") for t in set_text]).to(device)
        text_features = self.vlmodel.encode_text(text_inputs)
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)
        text_features_dict = {set_text[i]:text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict
    
    def __getitem__(self, index):
        
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject
        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        # eeg = torch.fft.fft(eeg, dim=1)
        # eeg = torch.abs(eeg)

        label = self.loaded_data[subject]['label'][trial_index]
        img = self.loaded_data[subject]['img'][trial_index]
        img_features = self.img_features[img]
        text =  f"This is a {self.loaded_data[subject]['text'][trial_index]}."
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]]
        session = self.loaded_data[subject]['session'][trial_index]
        
        if self.transform:
            eeg = self.transform(eeg)
        
        return eeg, label, img, img_features,text, text_features,session,subject

    def __len__(self):
        return self.trial_all_subjects


db_mapping = {
    'eeg_train': 0,
    'eeg_test': 1,
}
class EEGDatasetRedis():
    def __init__(self, data_root,subjects, mode,selected_ch=None,model_type = 'ViT-H-14',average=None):
        self.data_root = data_root
        self.img_directory = os.path.join(self.data_root,'Image_set',f'{mode}_images')
        self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        self.all_class_names.sort()
        
        self.redis = redis.Redis(host='localhost', port=6379, db=db_mapping[f"eeg_{mode}"])
        self.keys = self.redis.keys()
        
        self.mode = mode
        self.subjects = subjects
        
        self.n_cls = 1654 if self.mode=='train' else 200
        self.selected_ch = selected_ch
        self.ch_names = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                                            'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
                                            'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                                            'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
                                            'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                                            'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                                            'O1', 'Oz', 'O2']
        
        features_filename = os.path.join(self.data_root,'Preprocessed_data_250Hz',f"{model_type.replace('/','-')}_features_{mode}.pt")
        if os.path.exists(features_filename) :
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
            self.text_features = saved_features['text_features']
        
        self.all_text_features = torch.from_numpy(np.concatenate([self.text_features[k].unsqueeze(0) for k in self.all_class_names])).to(device)
        
        if mode =='test':
            self.all_image_features = torch.from_numpy(np.concatenate([self.img_features[k].unsqueeze(0) for k in self.img_features])).to(device)
        
    def __getitem__(self, index):
        key = self.redis.keys()[index].decode('utf-8')
        value = self.redis.hgetall(key)
        
        item = {}
        for k,v in value.items():
            k =k.decode('utf-8')
            if k == 'eeg':
                v = pickle.loads(v)
            else:
                v =v.decode('utf-8')
            
            if k in ['session','repeat_id','label']:
                v = int(v)
            item[k]=v

        
        item['img_features'] = self.img_features[item['img']]
        item['text_features'] = self.text_features[item['text']]
        
        return item['eeg'],item['label'],item['img'],item['img_features'],item['text'],item['text_features'],item['session']

    def __len__(self):
        return len(self.keys)

if __name__ == "__main__":
    # train_dataset = SingleEEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz',subject='sub-10',mode='train')
    # test_dataset = SingleEEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz',subject='sub-10',mode='test')
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    # train_dataset = EEGDatasetRedis(data_root='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small',subjects=['sub-10'],mode='train')
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    def set_logging(file_path):
        logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(message)s',filename=file_path,force=True)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger('').addHandler(console_handler)
    
    set_logging("tmp/tmp.LOG")
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    subjects = ['sub-08']
    
    test_dataset = EEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten',subjects=subjects,model_type='ViT-B/32',mode='test',avg=False)
    train_dataset = EEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten',subjects=subjects,model_type='ViT-B/32',mode='train',avg=False)
    
    test_dataset = EEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten',subjects=subjects,model_type='ViT-B/32',mode='test',avg=True)
    train_dataset = EEGDataset(data_dir='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz_whiten',subjects=subjects,model_type='ViT-B/32',mode='train',avg=True)
    
    # test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False, drop_last=False,num_workers=4)
    # train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    
    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # for epoch in range(1):
    #     for i,sample in enumerate(test_loader):
    #         eeg, label, img, img_features,text, text_features,session,subject = sample
    #         eeg = eeg.to(device)
            
    #     for i,sample in enumerate(train_loader):
    #         eeg, label, img, img_features,text, text_features,session,subject = sample
    #         eeg = eeg.to(device)
       