# train_advanced_cnn.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from models.basic_cnn import BasicSteelCNN

class AdvancedSteelDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=224):
        self.data_dir = data_dir
        self.transform = transform
        self.img_size = img_size
        self.images = []
        self.labels = []

        for class_name in ['crack', 'no_crack']:
            class_dir = os.path.join(data_dir, class_name)
            class_label = 1 if class_name == 'crack' else 0
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        self.images.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image=image)['image']
        else:
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0

        return image, label

def get_albumentations_transforms(img_size=224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    return train_transform, val_transform

def get_weighted_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_image, target_class=None):
        self.model.eval()
        output = self.model(input_image.unsqueeze(0))
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Ensure hooks are registered correctly.")

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[1]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, output

def train_advanced_cnn():
    config = {
        'img_size': 224,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'epochs': 40,
        'early_stop_patience': 7,
        'num_workers': 4
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform, val_transform = get_albumentations_transforms(config['img_size'])
    train_dataset = AdvancedSteelDataset('./dataset/train', transform=train_transform)
    val_dataset = AdvancedSteelDataset('./dataset/val', transform=val_transform)

    if len(train_dataset) == 0:
        print("No training data found. Check dataset structure.")
        return None, None

    train_sampler = get_weighted_sampler(train_dataset.labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model = BasicSteelCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [],
        'learning_rate': []
    }

    best_val_f1 = 0.0
    early_stop_counter = 0

    print("Starting advanced CNN training...")

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]} [Train]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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

        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)

        print(f'\nEpoch {epoch+1}/{config["epochs"]}:')
        print(f'  LR: {current_lr:.2e}')
        print(f'  Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%')
        print(f'  Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            early_stop_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'config': config,
                'gradcam_layer': 'conv4'
            }, 'checkpoints/best_advanced_cnn.pth')
            print(f'  ✓ New best model saved! F1: {val_f1:.4f}')
        else:
            early_stop_counter += 1

        if early_stop_counter >= config['early_stop_patience']:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

        scheduler.step()

    return model, history

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    model, history = train_advanced_cnn()
