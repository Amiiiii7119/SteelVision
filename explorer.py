# explorer.py
import os

def explore_directory():
    print("Directory exploration")
    print("=" * 60)
    print(f"Current directory: {os.getcwd()}")
    print("\nFolders and files:")
    print("-" * 60)

    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")

        sub_indent = " " * 2 * (level + 1)
        for file in files[:10]:  # show first 10 files
            print(f"{sub_indent}{file}")
        if len(files) > 10:
            print(f"{sub_indent}... {len(files) - 10} more files")

        # Limit depth to avoid large output
        if level >= 2:
            break

if __name__ == "__main__":
    explore_directory()
