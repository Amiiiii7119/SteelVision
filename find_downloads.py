# find_downloads.py
import os

def find_downloaded_dataset():
    """Search common download locations for the NEU dataset."""
    search_paths = [
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/Documents"),
        ".",
    ]

    print("Searching for NEU dataset...")

    defect_classes = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches"
    ]

    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"Checking: {search_path}")

            for item in os.listdir(search_path):
                item_path = os.path.join(search_path, item)

                # ZIP files
                if "neu" in item.lower() and item.endswith(".zip"):
                    print(f"Found ZIP: {item_path}")
                    return item_path

                # Extracted folders
                if any(key in item.lower() for key in ["neu", "surface", "defect"]):
                    if os.path.isdir(item_path):
                        sub_items = os.listdir(item_path)
                        found_classes = [s for s in sub_items if s in defect_classes]

                        if found_classes:
                            print(f"Found dataset folder: {item_path}")
                            print(f"Contains: {found_classes}")
                            return item_path

    print("Dataset not found in common locations.")
    return None

if __name__ == "__main__":
    dataset_path = find_downloaded_dataset()

    if dataset_path:
        print(f"\nFound dataset at: {dataset_path}")
        print("Next steps:")
        print("1. Move the folder or ZIP to your project directory.")
        print("2. Run: python complete_dataset_fixer.py")
    else:
        print("\nDataset not located.")
        print("Download from:")
        print("https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
