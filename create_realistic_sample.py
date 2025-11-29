# create_realistic_sample.py
import os
import cv2
import numpy as np
import random
from sklearn.model_selection import train_test_split

def create_realistic_steel_images():
    """Create realistic steel surface images for NEU-CLS."""
    print("Creating steel defect images...")

    base_dir = "NEU-CLS"
    classes = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    ]
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    STEEL_COLOR = [180, 180, 180]

    def create_crazing():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for _ in range(15):
            start_x, start_y = random.randint(10, 190), random.randint(10, 190)
            length = random.randint(20, 80)
            angle = random.uniform(0, 2 * np.pi)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            end_x = max(10, min(190, end_x))
            end_y = max(10, min(190, end_y))
            cv2.line(img, (start_x, start_y), (end_x, end_y), (80, 80, 80), 1)
        return img

    def create_inclusion():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for _ in range(random.randint(5, 15)):
            x, y = random.randint(20, 180), random.randint(20, 180)
            radius = random.randint(3, 10)
            darkness = random.randint(50, 120)
            cv2.circle(img, (x, y), radius, (darkness, darkness, darkness), -1)
        return img

    def create_patches():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for _ in range(random.randint(2, 6)):
            center_x, center_y = random.randint(30, 170), random.randint(30, 170)
            width, height = random.randint(15, 40), random.randint(15, 40)
            pts = []
            for i in range(6):
                angle = 2 * np.pi * i / 6
                radius_x = width // 2 + random.randint(-5, 5)
                radius_y = height // 2 + random.randint(-5, 5)
                x = center_x + int(radius_x * np.cos(angle))
                y = center_y + int(radius_y * np.sin(angle))
                pts.append((x, y))
            pts = np.array(pts, np.int32)
            color_variation = random.randint(-30, 30)
            patch_color = [
                STEEL_COLOR[0] + color_variation,
                STEEL_COLOR[1] + color_variation,
                STEEL_COLOR[2] + color_variation
            ]
            cv2.fillPoly(img, [pts], patch_color)
        return img

    def create_pitted_surface():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for _ in range(random.randint(20, 40)):
            x, y = random.randint(10, 190), random.randint(10, 190)
            radius = random.randint(1, 4)
            cv2.circle(img, (x, y), radius, (100, 100, 100), -1)
        for _ in range(8):
            start_x, start_y = random.randint(20, 180), random.randint(20, 180)
            length = random.randint(15, 40)
            angle = random.uniform(0, 2 * np.pi)
            end_x = int(start_x + length * np.cos(angle))
            end_y = int(start_y + length * np.sin(angle))
            end_x = max(20, min(180, end_x))
            end_y = max(20, min(180, end_y))
            cv2.line(img, (start_x, start_y), (end_x, end_y), (90, 90, 90), 1)
        return img

    def create_rolled_in_scale():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for i in range(5):
            y = 40 + i * 30
            cv2.line(img, (10, y), (190, y), (150, 150, 150), 2)
        for _ in range(random.randint(10, 25)):
            x, y = random.randint(15, 185), random.randint(15, 185)
            radius = random.randint(2, 6)
            shade = random.randint(130, 170)
            cv2.circle(img, (x, y), radius, (shade, shade, shade), -1)
        return img

    def create_scratches():
        img = np.full((200, 200, 3), STEEL_COLOR, dtype=np.uint8)
        for _ in range(random.randint(2, 4)):
            start_x, start_y = random.randint(20, 80), random.randint(20, 180)
            length = random.randint(60, 120)
            if random.random() > 0.3:
                end_x = start_x + length
                end_y = start_y + random.randint(-20, 20)
            else:
                end_x = start_x + random.randint(-20, 20)
                end_y = start_y + length
            end_x = max(20, min(180, end_x))
            end_y = max(20, min(180, end_y))
            cv2.line(img, (start_x, start_y), (end_x, end_y), (70, 70, 70), 2)
            for offset in (-2, 2):
                cv2.line(img, (start_x, start_y + offset), (end_x, end_y + offset), (90, 90, 90), 1)
        return img

    defect_functions = {
        "crazing": create_crazing,
        "inclusion": create_inclusion,
        "patches": create_patches,
        "pitted_surface": create_pitted_surface,
        "rolled-in_scale": create_rolled_in_scale,
        "scratches": create_scratches
    }

    images_per_class = 50
    for class_name, create_function in defect_functions.items():
        print(f"Creating images for: {class_name}")
        class_dir = os.path.join(base_dir, class_name)
        for i in range(images_per_class):
            img = create_function()
            cv2.imwrite(os.path.join(class_dir, f"{class_name}_{i:03d}.jpg"), img)

    print(f"Created {images_per_class} images for each class.")
    return True

def organize_for_steelvision():
    """Organize NEU-CLS into dataset/train and dataset/val."""
    print("Organizing dataset for SteelVision...")

    os.makedirs("dataset/train/crack", exist_ok=True)
    os.makedirs("dataset/train/no_crack", exist_ok=True)
    os.makedirs("dataset/val/crack", exist_ok=True)
    os.makedirs("dataset/val/no_crack", exist_ok=True)

    crack_classes = ["crazing", "scratches", "pitted_surface"]
    no_crack_classes = ["inclusion", "rolled-in_scale", "patches"]

    for class_name in crack_classes + no_crack_classes:
        class_path = os.path.join("NEU-CLS", class_name)
        images = [f for f in os.listdir(class_path) if f.endswith(".jpg")]
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
        target_class = "crack" if class_name in crack_classes else "no_crack"

        for img_name in train_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join("dataset", "train", target_class, f"{class_name}_{img_name}")
            os.rename(src, dst)

        for img_name in val_imgs:
            src = os.path.join(class_path, img_name)
            dst = os.path.join("dataset", "val", target_class, f"{class_name}_{img_name}")
            os.rename(src, dst)

    train_crack = len(os.listdir("dataset/train/crack"))
    train_no_crack = len(os.listdir("dataset/train/no_crack"))
    val_crack = len(os.listdir("dataset/val/crack"))
    val_no_crack = len(os.listdir("dataset/val/no_crack"))

    print("Dataset organized.")
    print(f"Training: crack={train_crack}, no_crack={train_no_crack}")
    print(f"Validation: crack={val_crack}, no_crack={val_no_crack}")
    return True

def verify_dataset():
    """Verify required dataset directories and counts."""
    print("Verifying dataset...")
    required_paths = [
        "dataset/train/crack",
        "dataset/train/no_crack",
        "dataset/val/crack",
        "dataset/val/no_crack"
    ]
    all_good = True
    for path in required_paths:
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if f.endswith(".jpg")])
            status = "OK" if count > 0 else "EMPTY"
            print(f"{path:30} - {status:6} - {count:3} images")
            if count == 0:
                all_good = False
        else:
            print(f"{path:30} - MISSING")
            all_good = False
    return all_good

if __name__ == "__main__":
    print("SteelVision sample dataset creator")
    print("=" * 50)

    if create_realistic_steel_images():
        if organize_for_steelvision():
            if verify_dataset():
                print("Ready to train. Suggested commands:")
                print("  python train_basic_cnn.py")
                print("  python train_advanced_cnn.py")
                print("  streamlit run steelvision_app.py")
            else:
                print("Dataset verification failed.")
        else:
            print("Organization failed.")
    else:
        print("Image creation failed.")
