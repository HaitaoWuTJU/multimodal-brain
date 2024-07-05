import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import open_clip
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
import os
from PIL import Image
import clip

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
vlmodel, preprocess, _ = open_clip.create_model_and_transforms('ViT-bigG-14',pretrained='laion2b_s39b_b160k', precision='fp32', device = device)
# vlmodel, preprocess = clip.load('ViT-L/14@336px', device=device)
class ThingsDataset():
    def __init__(self, data_root, mode='test'):
        self.data_root = data_root
        self.data_path =  os.path.join(self.data_root,f'{mode}_images')
        
        self.images_path = []  
        self.classes = []
        self.labels = []
        all_folders = [d for d in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, d))]
        all_folders.sort() 
        for folder in all_folders:
            folder_path = os.path.join(self.data_path, folder)
            
            class_text = folder.split('_',1)[-1].replace('_',' ')
            all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_images.sort()
            
            self.classes.append(class_text)
            self.labels.extend(class_text for img in all_images)
            self.images_path.extend(os.path.join(folder_path, img) for img in all_images)
        
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = preprocess(Image.open(image_path).convert('RGB'))
        label = self.labels[index]
        
        return image, label

    def __len__(self):
        return len(self.images_path)



if __name__=='__main__':
    
    
    test_dataset = ThingsDataset(data_root='/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Image_set',mode='test')
    classes = test_dataset.classes
    class_descriptions = [f"This is a {cls}." for cls in classes] 
    text_tokens = open_clip.tokenize(class_descriptions).to(device)
    # text_tokens = clip.tokenize(class_descriptions).to(device)
    test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)  # Adjust batch_size according to your needs
    results = []
    
    with torch.no_grad():
        text_features = vlmodel.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    def predict_class(images):
        images = images.to(device)
        with torch.no_grad():
            image_features = vlmodel.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        print(image_features.shape)
        similarity = (image_features @ text_features.T).cpu()
        predictions = similarity.argmax(dim=-1)
        return [classes[pred] for pred in predictions]
    
    all_predicted_classes = []
    all_true_labels = []
    for i,batch in tqdm(enumerate(test_dataloader)):
        images, labels = batch
        predicted_classes = predict_class(images)
        all_predicted_classes.extend(predicted_classes)
        all_true_labels.extend(labels) 
        
    correct_predictions = sum(1 for true, pred in zip(all_true_labels, all_predicted_classes) if true == pred)
    total_predictions = len(all_true_labels)
    accuracy = correct_predictions / total_predictions
    
    
    print(f"Accuracy: {accuracy:.3f}")
    
    for i,(true, pred) in enumerate(zip(all_true_labels, all_predicted_classes)):
        if true != pred:
            print(i)
    
    df = pd.DataFrame({
    'all_predicted_classes': all_predicted_classes,
    'all_true_labels': all_true_labels
    })
    df.to_csv('predictions_labels.tsv', sep='\t', index=False)