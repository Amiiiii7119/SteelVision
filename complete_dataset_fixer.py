# complete_dataset_fixer.py
import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split

def find_original_dataset():
    print("Searching for original dataset...")
    
    search_locations = [
        ".",
        "archive",
        "neu-surface-defect-database",
        "NEU Surface Defect Database",
        "NEU-CLS",
        "downloads",
        "dataset",
    ]
    
    zip_files = [f for f in os.listdir() if f.endswith(".zip")]
    if zip_files:
        print(f"Found ZIP files: {zip_files}")
    
    defect_classes = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    ]
    
    for location in search_locations:
        if os.path.exists(location):
            print(f"Checking: {location}")
            items = os.listdir(location)

            found_classes = [item for item in items if item in defect_classes]
            if found_classes:
                print(f"Found dataset in: {location}")
                return location

            for item in items:
                item_path = os.path.join(location, item)
                if os.path.isdir(item_path):
                    sub_items = os.listdir(item_path)
                    sub_found = [s for s in sub_items if s in defect_classes]
                    if sub_found:
                        print(f"Found dataset in: {location}/{item}")
                        return item_path
    
    print("Dataset not found.")
    return None


def extract_zip_if_needed():
    zip_files = [f for f in os.listdir() if f.endswith(".zip") and "neu" in f.lower()]
    
    for zip_file in zip_files:
        extract_folder = zip_file.replace(".zip", "")
        print(f"Found ZIP file: {zip_file}")
        
        if not os.path.exists(extract_folder):
            print(f"Extracting {zip_file}...")
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
                print(f"Extracted to: {extract_folder}")
                return extract_folder
            except Exception as e:
                print(f"Extraction failed: {e}")
    
    return None


def organize_from_source(source_path):
    crack_classes = ["crazing", "scratches", "pitted_surface"]
    no_crack_classes = ["inclusion", "rolled-in_scale", "patches"]

    print(f"Organizing dataset from: {source_path}")
    items = os.listdir(source_path)

    found_crack = [i for i in items if i in crack_classes]
    found_no_crack = [i for i in items if i in no_crack_classes]

    if not found_crack and not found_no_crack:
        for item in items:
            sub_path = os.path.join(source_path, item)
            if os.path.isdir(sub_path):
                sub_items = os.listdir(sub_path)
                has_crack = [s for s in sub_items if s in crack_classes]
                has_no_crack = [s for s in sub_items if s in no_crack_classes]
                if has_crack or has_no_crack:
                    return organize_from_source(sub_path)

    crack_images = []
    no_crack_images = []

    for cls in crack_classes + no_crack_classes:
        class_path = os.path.join(source_path, cls)
        if os.path.exists(class_path):
            images = [
                os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ]
            if cls in crack_classes:
                crack_images.extend([(img, cls) for img in images])
            else:
                no_crack_images.extend([(img, cls) for img in images])

    if not crack_images and not no_crack_images:
        print("No images found in dataset.")
        return False

    dataset_dir = "dataset"
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    for split_dir in [train_dir, val_dir]:
        for cls in ["crack", "no_crack"]:
            os.makedirs(os.path.join(split_dir, cls), exist_ok=True)

    crack_train, crack_val = train_test_split(crack_images, test_size=0.2, random_state=42)
    no_crack_train, no_crack_val = train_test_split(no_crack_images, test_size=0.2, random_state=42)

    def copy_images(image_list, destination, target_class):
        for img_path, original_class in image_list:
            img_name = os.path.basename(img_path)
            new_name = f"{original_class}_{img_name}"
            dest_path = os.path.join(destination, target_class, new_name)
            shutil.copy2(img_path, dest_path)

    print("Copying images...")
    copy_images(crack_train, train_dir, "crack")
    copy_images(no_crack_train, train_dir, "no_crack")
    copy_images(crack_val, val_dir, "crack")
    copy_images(no_crack_val, val_dir, "no_crack")

    print("Dataset organized successfully.")
    print(f"Training: crack {len(crack_train)}, no_crack {len(no_crack_train)}")
    print(f"Validation: crack {len(crack_val)}, no_crack {len(no_crack_val)}")

    return True


def main():
    print("STEELVISION DATASET FIXER")
    print("=" * 50)

    extracted_path = extract_zip_if_needed()
    if extracted_path:
        organize_from_source(extracted_path)
        return

    source_path = find_original_dataset()
    if source_path:
        organize_from_source(source_path)
    else:
        print("Dataset not found. Manual action required.")
        print("Look for folders: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches")
        print("Place them in this directory and run again.")
        print("Current directory:", os.getcwd())
        print("Contents:", os.listdir())


if __name__ == "__main__":
    main()
