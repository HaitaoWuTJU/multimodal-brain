import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score


def train_linear_probing(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10
    num_classes = 10 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=32, pin_memory=True)
    test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=32, pin_memory=True)
    
    base_model = models.resnet18(pretrained=True)
    for param in base_model.parameters():
        param.requires_grad = False  # freeze

    # linear layer
    base_model.fc = nn.Linear(base_model.fc.in_features, num_classes).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!")
        base_model = nn.DataParallel(base_model)
    base_model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.module.fc.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        loss = train_linear_probing(base_model, train_loader, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}')
        
        accuracy = evaluate_model(base_model, test_loader, device)
        print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()