import torch,os
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, subject_id=1,data_root='/root/multimodal_brain/datasets/things-eeg'):
        self.subject_id = subject_id
        self.eeg_data_path = os.path.join(data_root,f'sub-{subject_id:02}','eeg')

        print(self.eeg_data_path)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return sample

if __name__=='__main__':
    dataset=CustomDataset()