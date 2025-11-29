# enhanced_predict.py
import argparse
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

def enhance_crack_visibility(image_path):
    """Preprocess image and return multiple enhanced variants."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    enhanced_images = []

    # Original
    enhanced_images.append(image)

    # CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0).apply(lab[:, :, 0])
    clahe_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhanced_images.append(clahe_enhanced)

    # Edge enhancement
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    edge_enhanced = cv2.addWeighted(image, 0.7, edges_rgb, 0.3, 0)
    enhanced_images.append(edge_enhanced)

    # Sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    enhanced_images.append(sharpened)

    return enhanced_images

def predict_with_enhancement(image_path, model_path='checkpoints/best_basic_cnn.pth'):
    """Predict using multiple enhanced versions and return ensemble result."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    from models.basic_cnn import BasicSteelCNN
    model = BasicSteelCNN(num_classes=2)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    enhanced_versions = enhance_crack_visibility(image_path)

    predictions = []
    probabilities = []

    for enhanced_img in enhanced_versions:
        pil_img = Image.fromarray(enhanced_img)
        tensor = transform(pil_img)
        input_tensor = tensor.unsqueeze(0).to(device)  # type: ignore

        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            pred_class = torch.argmax(output, dim=1).item()

            predictions.append(pred_class)
            probabilities.append(prob[0, 1].item())  # crack probability

    # Ensemble: majority vote + average probability
    final_prediction = 1 if sum(predictions) >= len(predictions) / 2 else 0
    avg_probability = sum(probabilities) / len(probabilities)

    print(f"Image: {image_path}")
    print(f"Individual predictions: {predictions}")
    print(f"Individual probabilities: {[f'{p:.3f}' for p in probabilities]}")
    print(f"Final prediction: {'CRACK' if final_prediction == 1 else 'NO CRACK'}")
    print(f"Confidence: {avg_probability:.3f}")

    return final_prediction, avg_probability

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='checkpoints/best_basic_cnn.pth', help='Model path')
    args = parser.parse_args()
    predict_with_enhancement(args.image, args.model)
