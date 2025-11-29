# train_basic_cnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from models.basic_cnn import BasicSteelCNN

class SteelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        crack_dir = os.path.join(data_dir, 'crack')
        if os.path.exists(crack_dir):
            for img_name in os.listdir(crack_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(crack_dir, img_name))
                    self.labels.append(1)  # Crack = 1

        no_crack_dir = os.path.join(data_dir, 'no_crack')
        if os.path.exists(no_crack_dir):
            for img_name in os.listdir(no_crack_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.images.append(os.path.join(no_crack_dir, img_name))
                    self.labels.append(0)  # No crack = 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_basic_cnn():
    img_size = 224
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SteelDataset('./dataset/train', transform=train_transform)
    val_dataset = SteelDataset('./dataset/val', transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BasicSteelCNN(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    best_val_acc = 0.0

    print("Starting Basic CNN Training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * train_correct / train_total:.2f}%'
            })

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        val_precision = precision_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
        val_recall = recall_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0
        val_f1 = f1_score(all_labels, all_preds, zero_division=0) if all_labels else 0.0

        history['train_loss'].append(train_loss/len(train_loader) if len(train_loader) > 0 else 0.0)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss/len(val_loader) if len(val_loader) > 0 else 0.0)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)

        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        print(f'  Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, 'checkpoints/best_basic_cnn.pth')
            print(f'  ✓ New best model saved! Val Acc: {val_acc:.2f}%')

        scheduler.step()

    plot_training_history(history)
    return model, history

def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0,0].plot(history['train_loss'], label='Train Loss')
    axes[0,0].plot(history['val_loss'], label='Val Loss')
    axes[0,0].set_title('Training and Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()

    axes[0,1].plot(history['train_acc'], label='Train Acc')
    axes[0,1].plot(history['val_acc'], label='Val Acc')
    axes[0,1].set_title('Training and Validation Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy (%)')
    axes[0,1].legend()

    axes[1,0].plot(history['val_precision'], label='Precision')
    axes[1,0].plot(history['val_recall'], label='Recall')
    axes[1,0].plot(history['val_f1'], label='F1-Score')
    axes[1,0].set_title('Validation Metrics')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Score')
    axes[1,0].legend()

    plt.tight_layout()
    plt.savefig('results/basic_cnn_training.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    model, history = train_basic_cnn()
