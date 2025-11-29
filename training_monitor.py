# train_advanced_cnn_fixed.py
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

import timm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class AdvancedSteelDataset(Dataset):
    def __init__(self, data_dir, transform=None, img_size=224):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.img_size = img_size
        self.images = []
        self.labels = []

        class_map = {'no_crack': 0, 'crack': 1}
        for cls_name, label in class_map.items():
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                continue
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.images.append(str(p))
                    self.labels.append(label)

        assert len(self.images) == len(self.labels), "Images and labels length mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

def get_transforms(img_size=224):
    train_transform = A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.RandomResizedCrop(img_size, img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.OneOf([A.MotionBlur(blur_limit=3), A.MedianBlur(blur_limit=3), A.GaussianBlur(blur_limit=3)], p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

def create_weighted_sampler(labels):
    labels = np.array(labels)
    class_counts = np.bincount(labels)
    if len(class_counts) < 2:
        class_counts = np.pad(class_counts, (0, 2 - len(class_counts)), constant_values=1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights.tolist(), num_samples=len(sample_weights), replacement=True)
    return sampler, class_weights

def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, None, None, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, index, lam

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_module.register_forward_hook(forward_hook)
        if hasattr(self.target_module, "register_full_backward_hook"):
            self.target_module.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
        else:
            self.target_module.register_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.unsqueeze(0)
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check if hooks are registered correctly.")
        grads = self.gradients.cpu().numpy()[0]
        acts = self.activations.cpu().numpy()[0]
        weights = np.mean(grads, axis=(1, 2))
        cam = np.sum(weights[:, None, None] * acts, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def train_advanced_cnn(
    train_dir='./dataset/train',
    val_dir='./dataset/val',
    img_size=224,
    batch_size=24,
    epochs=30,
    lr=3e-4,
    weight_decay=1e-4,
    mixup_alpha=0.3,
    device=None
):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print("Device:", device)

    train_tf, val_tf = get_transforms(img_size)
    train_ds = AdvancedSteelDataset(train_dir, transform=train_tf, img_size=img_size)
    val_ds = AdvancedSteelDataset(val_dir, transform=val_tf, img_size=img_size)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Train or validation dataset is empty. Check paths and structure.")

    sampler, class_weights = create_weighted_sampler(train_ds.labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.1)

    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    best_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            labels = labels.to(device)

            mixed_x = images
            y_a = labels
            y_b = labels
            lam = 1.0
            use_mixup = False

            if mixup_alpha and mixup_alpha > 0:
                mixed_x, y_a, y_b, _, lam = mixup_data(images, labels, alpha=mixup_alpha)
                if y_b is not None:
                    mixed_x = mixed_x.to(device)
                    y_a = y_a.to(device)
                    y_b = y_b.to(device)
                    use_mixup = True

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(mixed_x if use_mixup else images)
                    if use_mixup:
                        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                    else:
                        loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(mixed_x if use_mixup else images)
                if use_mixup:
                    loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                else:
                    loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            running_loss += loss.item()
            scheduler.step()

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                labels = labels.to(device)
                if scaler:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        print(f"Epoch {epoch+1}: Train loss {avg_train_loss:.4f} | Val loss {avg_val_loss:.4f} | Prec {val_precision:.4f} | Rec {val_recall:.4f} | F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'class_weights': class_weights
            }, 'checkpoints/best_advanced_cnn_fixed.pth')
            print("Saved new best model.")

    return model, history

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    model, history = train_advanced_cnn(
        train_dir='./dataset/train',
        val_dir='./dataset/val',
        img_size=320,
        batch_size=16,
        epochs=50,
        lr=5e-4,
        weight_decay=1e-4,
        mixup_alpha=0.2
    )
    print("Training complete.")