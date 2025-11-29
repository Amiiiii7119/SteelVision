# verify_cracks.py
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def analyze_crack_visibility(image_path):
    """Analyze whether cracks are visible in the image."""
    print(f"Analyzing: {os.path.basename(image_path)}")

    image = cv2.imread(image_path)
    if image is None:
        print("Could not load image")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Edge density
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

    # Contour analysis
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crack_like_contours = 0
    total_contour_length = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            # Long and thin contours are more likely cracks
            if circularity < 0.1 and perimeter > 20:
                crack_like_contours += 1
                total_contour_length += perimeter

    print("Image Analysis:")
    print(f"  - Edge density: {edge_density:.4f}")
    print(f"  - Crack-like contours: {crack_like_contours}")
    print(f"  - Total crack length (px): {total_contour_length:.2f}")

    # Simple verdict rules
    if edge_density > 0.01 and crack_like_contours > 2:
        print("VERDICT: LIKELY CRACK PRESENT")
    elif edge_density > 0.005 and crack_like_contours > 1:
        print("VERDICT: POSSIBLE CRACK")
    else:
        print("VERDICT: UNLIKELY TO HAVE VISIBLE CRACK")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title(f'Edges (density: {edge_density:.4f})')
    axes[1].axis('off')

    contour_img = image_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (255, 0, 0), 2)
    axes[2].imshow(contour_img)
    axes[2].set_title(f'Contours: {crack_like_contours} crack-like')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        analyze_crack_visibility(sys.argv[1])
    else:
        test_images = []
        if os.path.exists('dataset/val/crack'):
            test_images = [os.path.join('dataset/val/crack', f) for f in os.listdir('dataset/val/crack')[:3]]
        for img_path in test_images:
            analyze_crack_visibility(img_path)
            print("-" * 50)
