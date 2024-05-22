import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from eegdatasets_leaveone import EEGDataset

def accuracy(output, target, topk=(1,)):
    """Top-k acc"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1024, 512),  
            nn.ReLU(),      
            nn.Linear(512, 256),  
            nn.ReLU(),         
            nn.Linear(256, 27)
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Use {torch.cuda.device_count()} GPUs!")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

data_path = "/root/workspace/wht/multimodal_brain/datasets/things-eeg-small/Preprocessed_data_250Hz"
train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True)    
test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)


num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        x, label, text, text_features, img, img_features = data
        img_features = img_features.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        outputs = model(img_features)
        
        print(outputs.shape)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    top1_acc = []
    top5_acc = []
    with torch.no_grad():
        for data in test_loader:
            x, label, text, text_features, img, img_features = data
            img_features = img_features.to(device)
            label = label.to(device)
            outputs = model(img_features)
            acc1, acc5 = accuracy(outputs, label, topk=(1, 5))
            top1_acc.append(acc1)
            top5_acc.append(acc5)

    avg_top1_acc = torch.cat(top1_acc).mean().item()
    avg_top5_acc = torch.cat(top5_acc).mean().item()
    print(f'Test Top-1 Accuracy: {avg_top1_acc:.2f}%, Test Top-5 Accuracy: {avg_top5_acc:.2f}%')