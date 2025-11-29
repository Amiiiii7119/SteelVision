# download_helper.py
import os
import zipfile

def provide_download_instructions():
    print("Manual download instructions:")
    print("=" * 50)
    print("1. Go to: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database")
    print("2. Click the 'Download' button (requires Kaggle account)")
    print("3. Save the ZIP file to your SteelVision project folder")
    print("4. Expected filename: 'neu-surface-defect-database.zip'")
    print("\nAlternative: Kaggle API download (requires setup):")
    print("   https://www.kaggle.com/api/v1/datasets/download/kaustubhdikshit/neu-surface-defect-database")
    print("\nAfter downloading, run: python complete_dataset_fixer.py")

def check_and_extract():
    """Check for the NEU ZIP file and extract it to 'NEU-CLS'."""
    zip_files = [f for f in os.listdir() if f.endswith(".zip") and "neu" in f.lower()]

    if zip_files:
        zip_file = zip_files[0]
        print(f"Found ZIP file: {zip_file}")
        extract_folder = "NEU-CLS"

        if not os.path.exists(extract_folder):
            print("Extracting...")
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_folder)
                print(f"Extracted to: {extract_folder}")
                return True
            except Exception as e:
                print(f"Extraction failed: {e}")
                return False
        else:
            print("Already extracted.")
            return True
    else:
        print("No dataset ZIP file found.")
        provide_download_instructions()
        return False

if __name__ == "__main__":
    check_and_extract()
