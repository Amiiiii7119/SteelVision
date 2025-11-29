# check_training_results.py
import torch
import os

def check_training_success():
    print("CHECKING TRAINING RESULTS")
    print("=" * 50)
    
    model_files = []
    if os.path.exists("checkpoints"):
        model_files = [f for f in os.listdir("checkpoints") if f.endswith(".pth")]
    
    if model_files:
        print("Found trained models:")
        for model_file in model_files:
            file_path = os.path.join("checkpoints", model_file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            try:
                checkpoint = torch.load(file_path, map_location="cpu")
                val_acc = checkpoint.get("val_acc", "Unknown")
                val_f1 = checkpoint.get("val_f1", "Unknown")
                print(f"  {model_file}")
                print(f"     Size: {size_mb:.1f} MB | Accuracy: {val_acc}% | F1: {val_f1}")
            except:
                print(f"  {model_file} - {size_mb:.1f} MB (Could not load details)")
    else:
        print("No trained models found")
        print("Run: python train_basic_cnn.py")
    
    if os.path.exists("results"):
        result_files = os.listdir("results")
        if result_files:
            print(f"\nTraining graphs: {result_files}")
        else:
            print("\nNo result graphs found")

if __name__ == "__main__":
    check_training_success()
