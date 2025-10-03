from pathlib import Path

DATASET_DIR = Path("dataset/merged")

def inspect_dataset():
    total = 0
    print(f"Inspecting: {DATASET_DIR}\n")
    for cls_dir in sorted(DATASET_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.*")))
            print(f"{cls_dir.name:<10}: {count}")
            total += count
    print("\nTotal images:", total)

if __name__ == "__main__":
    inspect_dataset()
