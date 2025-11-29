# predict.py
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.basic_cnn import BasicSteelCNN
from train_advanced_cnn import GradCAM


class SteelVisionPredictor:
    def __init__(self, checkpoint_path, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model, self.checkpoint = self.load_model(checkpoint_path)
        self.transform = self.get_transform()

    def load_model(self, checkpoint_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = BasicSteelCNN(num_classes=2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        val_acc = checkpoint.get('val_acc')
        val_f1 = checkpoint.get('val_f1')

        print(f"Model loaded from {checkpoint_path}")
        if val_acc is not None:
            print(f"Validation accuracy: {val_acc:.2f}%")
        else:
            print("Validation accuracy: N/A")

        if val_f1 is not None:
            print(f"Validation F1: {val_f1:.4f}")
        else:
            print("Validation F1: N/A")

        return model, checkpoint

    def get_transform(self):
        """Image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path, generate_heatmap=True):
        """Predict a single image and optionally generate Grad-CAM heatmap"""
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        transformed: torch.Tensor = self.transform(image)  # type: ignore
        input_tensor = transformed.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class_idx: int = int(torch.argmax(outputs, dim=1).item())
            confidence = probabilities[0, predicted_class_idx].item()

        heatmap = None
        overlay = None
        if generate_heatmap:
            gradcam = GradCAM(self.model, self.model.conv4)
            heatmap, _ = gradcam.generate_heatmap(input_tensor[0], target_class=predicted_class_idx)

            # Resize heatmap to original image size and create overlay
            heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(np.ascontiguousarray(heatmap_uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(original_image, 0.7, heatmap_colored, 0.3, 0)

        results = {
            'predicted_class': predicted_class_idx,
            'class_name': 'crack' if predicted_class_idx == 1 else 'no_crack',
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'heatmap': heatmap,
            'overlay': overlay,
            'original_image': original_image
        }

        return results

    def visualize_results(self, results, save_path=None):
        """Visualize prediction results and optional heatmap/overlay"""
        cols = 3
        fig, axes = plt.subplots(1, cols, figsize=(15, 5))

        # Original image
        axes[0].imshow(results['original_image'])
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Heatmap
        if results['heatmap'] is not None:
            axes[1].imshow(results['heatmap'], cmap='jet')
            axes[1].set_title('Grad-CAM Heatmap')
        else:
            axes[1].text(0.5, 0.5, 'No heatmap', ha='center', va='center')
        axes[1].axis('off')

        # Overlay
        if results['overlay'] is not None:
            # overlay may be in RGB order already; convert if needed
            try:
                axes[2].imshow(cv2.cvtColor(results['overlay'], cv2.COLOR_BGR2RGB))
            except Exception:
                axes[2].imshow(results['overlay'])
            axes[2].set_title('Heatmap Overlay')
        else:
            axes[2].text(0.5, 0.5, 'No overlay', ha='center', va='center')
        axes[2].axis('off')

        class_name = results['class_name'].upper()
        confidence = results['confidence']
        plt.suptitle(f'Prediction: {class_name} (Confidence: {confidence:.2%})', fontsize=16, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Results saved to {save_path}")

        plt.show()

    def batch_predict(self, image_dir, output_dir=None):
        """Predict multiple images in a directory (no heatmap generation)"""
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)
                       if f.lower().endswith(image_extensions)]

        results = []
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path, generate_heatmap=False)
                result['image_path'] = image_path
                results.append(result)
                print(f"{os.path.basename(image_path)}: {result['class_name']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

        total_count = len(results)
        crack_count = sum(1 for r in results if r['predicted_class'] == 1)

        if total_count > 0:
            print("\nBatch Prediction Summary:")
            print(f"  Total images: {total_count}")
            print(f"  Crack detected: {crack_count}")
            print(f"  No crack: {total_count - crack_count}")
            print(f"  Crack percentage: {100 * crack_count / total_count:.2f}%")
        else:
            print("No images processed in batch.")

        return results


def main():
    parser = argparse.ArgumentParser(description='SteelVision Crack Detection Predictor')
    parser.add_argument('--image', type=str, required=True, help='Path to input image or image directory (for batch)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_advanced_cnn.pth', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, help='Output path for results visualization')
    parser.add_argument('--batch', action='store_true', help='Batch prediction on directory')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')
    args = parser.parse_args()

    predictor = SteelVisionPredictor(args.checkpoint)

    if args.batch:
        predictor.batch_predict(args.image)
    else:
        results = predictor.predict_image(args.image)
        print("\nPrediction Results:")
        print(f"  Image: {args.image}")
        print(f"  Prediction: {results['class_name'].upper()}")
        print(f"  Confidence: {results['confidence']:.2%}")
        print(f"  Probabilities: [No Crack: {results['probabilities'][0]:.3f}, Crack: {results['probabilities'][1]:.3f}]")

        if not args.no_viz:
            output_path = args.output or 'prediction_results.png'
            predictor.visualize_results(results, save_path=output_path)


if __name__ == '__main__':
    main()
