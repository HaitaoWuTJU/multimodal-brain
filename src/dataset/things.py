import os
import pandas as pd
from PIL import Image
import torch
import open_clip

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ThingsDataset():
    def __init__(self, data_root, mode,transform):
        self.data_root = data_root
        self.mode = mode
        self.transform =transform
        
        self.img_dir =  os.path.join(self.data_root,'THINGS','Images')
        df = pd.read_csv(os.path.join(self.data_root,'THINGS',f'data_split_{self.mode}.csv'))
        self.df = df[df['category'] != '-1']
        
        self.label_map= {'food':0,'animals':1,'tool':2}
        
        model_type = 'ViT-H-14'
        features_filename = os.path.join(self.data_root,f'{model_type}_features_{mode}.pt')
        if os.path.exists(features_filename) :
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
        else:
            self.vlmodel, self.preprocess_train, self.feature_extractor = open_clip.create_model_and_transforms(
                model_type, pretrained='laion2b_s32b_b79k', precision='fp32', device = device)
            self.img_features = self.ImageEncoder(self.df['Img'].tolist())
            torch.save({
                'img_features': self.img_features,
            }, features_filename)
       
    @torch.no_grad()     
    def ImageEncoder(self,img_paths):
        batch_size = 64
        image_features_list = []
        for i in range(0, len(img_paths), batch_size):
            batch_images = img_paths[i:i + batch_size]
            image_inputs = torch.stack([self.preprocess_train(Image.open(os.path.join(self.img_dir,image_path.rsplit('_',1)[0],image_path)).convert('RGB')) for image_path in batch_images]).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            batch_image_features = batch_image_features/batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features.cpu())
        image_features = torch.cat(image_features_list, dim=0)
        return image_features
    
    def __getitem__(self, index):
        image_path = self.df.loc[index, 'Img']
        image = Image.open(os.path.join(self.img_dir,image_path.rsplit('_',1)[0],image_path)).convert('RGB')
        label_str = self.df.loc[index,'category']
        label = self.label_map[label_str]
        
        image_feature = self.img_features[index]
        
        if self.transform:
            image = self.transform(image)
        return image, label, image_feature

    def __len__(self):
        return len(self.df)

if __name__=="__main__":
    d=ThingsDataset('/root/workspace/wht/multimodal_brain/datasets/things','train')