# emergency_crack_fix.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

class EmergencyCrackDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.images = []
        self.labels = []

        print("Loading dataset...")

        # Load CRACK images with priority
        crack_images = []
        crack_dir = os.path.join(data_dir, 'crack')
        if os.path.exists(crack_dir):
            crack_images = [
                os.path.join(crack_dir, f)
                for f in os.listdir(crack_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            self.images.extend(crack_images)
            self.labels.extend([1] * len(crack_images))  # 1 for crack
            print(f"Loaded {len(crack_images)} crack images")

        # Load NO_CRACK images
        no_crack_dir = os.path.join(data_dir, 'no_crack')
        if os.path.exists(no_crack_dir):
            no_crack_images = [
                os.path.join(no_crack_dir, f)
                for f in os.listdir(no_crack_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            # Use fewer no_crack images to balance focus on cracks
            if mode == 'train':
                no_crack_images = no_crack_images[:len(crack_images)]
            self.images.extend(no_crack_images)
            self.labels.extend([0] * len(no_crack_images))  # 0 for no_crack
            print(f"Loaded {len(no_crack_images)} no_crack images")

        print(f"Total: {len(self.images)} images ({sum(self.labels)} cracks, {len(self.labels) - sum(self.labels)} no-cracks)")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]

            # Extreme augmentation for crack images during training
            if label == 1 and self.mode == 'train':
                image = self.extreme_crack_augmentation(np.array(image))
                image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_img = torch.randn(3, 224, 224)
            return dummy_img, self.labels[idx]

    def extreme_crack_augmentation(self, image):
        """Apply strong augmentations to emphasize cracks."""
        # Contrast enhancement (LAB CLAHE)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=4.0).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # Edge enhancement
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 10, 100)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        image = cv2.addWeighted(image, 0.6, edges_rgb, 0.4, 0)

        # Sharpening
        kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, kernel)

        return image

def emergency_retrain():
    print("Emergency crack detection retraining")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.7),
        transforms.RandomVerticalFlip(p=0.7),
        transforms.RandomRotation(degrees=25),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = EmergencyCrackDataset('./dataset/train', transform=train_transform, mode='train')
    val_dataset = EmergencyCrackDataset('./dataset/val', transform=val_transform, mode='val')

    if len(train_dataset) == 0:
        print("No training data found. Check dataset structure.")
        return

    # Heavy class weighting to prioritize crack detection
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_dataset.labels),
        y=train_dataset.labels
    )
    print(f"Class weights: {class_weights}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    from models.basic_cnn import BasicSteelCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BasicSteelCNN(num_classes=2).to(device)

    # Emphasize crack class in loss
    crack_weight = 5.0
    class_weights_tensor = torch.tensor([1.0, crack_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)

    num_epochs = 25
    best_crack_acc = 0.0
    patience = 5
    patience_counter = 0

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_crack_correct = 0
        train_crack_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            crack_mask = labels == 1
            if crack_mask.sum() > 0:
                _, predicted = torch.max(outputs, 1)
                train_crack_correct += ((predicted == labels) & crack_mask).sum().item()
                train_crack_total += crack_mask.sum().item()

        model.eval()
        val_crack_correct = 0
        val_crack_total = 0
        val_total_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                crack_mask = labels == 1
                if crack_mask.sum() > 0:
                    val_crack_correct += ((predicted == labels) & crack_mask).sum().item()
                    val_crack_total += crack_mask.sum().item()

                val_total_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        train_crack_acc = 100.0 * train_crack_correct / train_crack_total if train_crack_total > 0 else 0.0
        val_crack_acc = 100.0 * val_crack_correct / val_crack_total if val_crack_total > 0 else 0.0
        val_overall_acc = 100.0 * val_total_correct / val_total if val_total > 0 else 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"   Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"   Train Crack Acc: {train_crack_acc:.1f}%")
        print(f"   Val Crack Acc: {val_crack_acc:.1f}%")
        print(f"   Val Overall Acc: {val_overall_acc:.1f}%")

        # Save best model based on crack accuracy
        if val_crack_acc > best_crack_acc:
            best_crack_acc = val_crack_acc
            patience_counter = 0

            torch.save({
                'model_state_dict': model.state_dict(),
                'val_crack_acc': val_crack_acc,
                'val_overall_acc': val_overall_acc,
                'epoch': epoch,
                'emergency_trained': True
            }, 'checkpoints/emergency_crack_model.pth')
            print(f"New best crack accuracy: {val_crack_acc:.1f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("\nEmergency training completed.")
    print(f"Best crack detection accuracy: {best_crack_acc:.1f}%")

    if best_crack_acc > 70:
        print("Crack detection improved.")
    else:
        print("Crack detection may need more data or tuning.")

if __name__ == "__main__":
    emergency_retrain()
