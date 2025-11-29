"""
test_model.py

Test / inference script for the EfficientNet-based crack detection model.

Usage examples:
  # Evaluate on a test folder with structure test/crack, test/no_crack
  python test_model.py --checkpoint checkpoints/best_advanced_cnn_fixed.pth --test_dir ./dataset/test --img_size 320 --batch_size 16 --save_csv results/test_predictions.csv

  # Single image inference + optional Grad-CAM visualization
  python test_model.py --checkpoint checkpoints/best_advanced_cnn_fixed.pth --image ./samples/sample1.jpg --img_size 320 --gradcam --gradcam_out results/gradcam_sample1.png

Requirements:
  pip install torch torchvision timm albumentations opencv-python scikit-learn pandas tqdm
"""

import os
import argparse
from pathlib import Path
import csv

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import pandas as pd

class InferenceDataset(Dataset):
    def __init__(self, root, img_size=320, transform=None):
        """
        root: path to folder with subfolders for classes (e.g., root/crack, root/no_crack)
              or a single folder of images (labels will be None)
        """
        self.root = Path(root)
        self.img_paths = []
        self.labels = []
        self.img_size = img_size

        class_map = {'no_crack': 0, 'crack': 1}
        if any((self.root / cn).exists() for cn in class_map):
            for cls_name, label in class_map.items():
                cls_dir = self.root / cls_name
                if not cls_dir.exists():
                    continue
                for p in sorted(cls_dir.iterdir()):
                    if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                        self.img_paths.append(str(p))
                        self.labels.append(label)
        else:
            for p in sorted(self.root.iterdir()):
                if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    self.img_paths.append(str(p))
            self.labels = None

        if transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        image = cv2.imread(p)
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        data = self.transform(image=image)
        img = data['image']
        if self.labels is None:
            return img, p
        else:
            label = self.labels[idx]
            return img, label, p

class GradCAM:
    def __init__(self, model, target_module):
        self.model = model
        self.target_module = target_module
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_module.register_forward_hook(forward_hook)
        if hasattr(self.target_module, "register_full_backward_hook"):
            self.target_module.register_full_backward_hook(lambda m, gi, go: backward_hook(m, gi, go))
        else:
            self.target_module.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        input_tensor: torch.Tensor (C,H,W) on same device as model
        returns: heatmap (H, W) normalized to [0,1]
        """
        self.model.eval()
        x = input_tensor.unsqueeze(0)
        out = self.model(x)
        if target_class is None:
            target_class = out.argmax(dim=1).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations not captured. Check target_module selection.")

        grads = self.gradients.cpu().numpy()[0]    # (C, H, W)
        acts = self.activations.cpu().numpy()[0]   # (C, H, W)
        weights = np.mean(grads, axis=(1, 2))      # (C,)
        cam = np.sum(weights[:, None, None] * acts, axis=0)
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[1]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def load_checkpoint(checkpoint_path, device, model_name='efficientnet_b0', num_classes=2):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt

def evaluate(model, test_dir, device, img_size=320, batch_size=16, save_csv=None):
    ds = InferenceDataset(test_dir, img_size=img_size)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    y_true = []
    y_pred = []
    paths = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            imgs, labels, batch_paths = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = softmax(outputs)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            paths.extend(batch_paths)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print("Confusion matrix:\n", cm)
    print("Classification report:\n", report)

    if save_csv:
        df = pd.DataFrame({'path': paths, 'true': y_true, 'pred': y_pred})
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)
        print(f"Saved predictions to {save_csv}")

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'confusion_matrix': cm, 'report': report}

def infer_image(model, image_path, device, img_size=320, gradcam=False, gradcam_out=None):
    transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)), ToTensorV2()])
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    data = transform(image=img_rgb)
    tensor = data['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        pred = int(probs.argmax(dim=1).item())
        prob_val = probs[0, pred].item()

    result = {'path': image_path, 'pred': int(pred), 'prob': float(prob_val)}

    if gradcam:
        # Try to select a convolutional target module
        target_module = None
        for name, module in model.named_modules():
            if 'conv_head' in name or ('conv' in name and module.__class__.__name__.lower().startswith('conv')):
                target_module = module
        if target_module is None:
            target_module = list(model.modules())[-1]

        cam = GradCAM(model, target_module)
        heatmap = cam.generate(tensor.squeeze(0).cpu())

        H, W = img_rgb.shape[:2]
        heatmap_resized = cv2.resize((heatmap * 255).astype('uint8'), (W, H))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)

        if gradcam_out:
            Path(gradcam_out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(gradcam_out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        result['gradcam_out'] = gradcam_out

    return result

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint .pth')
    p.add_argument('--test_dir', type=str, default=None, help='Path to test folder (contains crack/no_crack) for batch evaluation')
    p.add_argument('--image', type=str, default=None, help='Single image path for inference')
    p.add_argument('--img_size', type=int, default=320)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--device', type=str, default=None, help='cpu or cuda (auto if not provided)')
    p.add_argument('--save_csv', type=str, default=None, help='Path to save csv of predictions when using --test_dir')
    p.add_argument('--gradcam', action='store_true', help='Generate Grad-CAM for single image inference')
    p.add_argument('--gradcam_out', type=str, default=None, help='Path to save gradcam overlay (single image use)')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model, ckpt = load_checkpoint(args.checkpoint, device)
    print("Loaded checkpoint.")

    if args.test_dir:
        metrics = evaluate(model, args.test_dir, device, img_size=args.img_size, batch_size=args.batch_size, save_csv=args.save_csv)
        print(f"Evaluation complete. F1: {metrics['f1']:.4f}")

    if args.image:
        res = infer_image(model, args.image, device, img_size=args.img_size, gradcam=args.gradcam, gradcam_out=args.gradcam_out)
        print("Inference result:", res)

if __name__ == '__main__':
    main()
