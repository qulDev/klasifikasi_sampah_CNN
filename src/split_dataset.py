import shutil
import random
from pathlib import Path

random.seed(42)

DATASET_DIR = Path("dataset/merged")
OUTPUT_DIR = Path("dataset/split")

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

def split_dataset():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    for split in ["train", "val", "test"]:
        for cls in DATASET_DIR.iterdir():
            (OUTPUT_DIR / split / cls.name).mkdir(parents=True, exist_ok=True)

    for cls in DATASET_DIR.iterdir():
        if cls.is_dir():
            images = list(cls.glob("*.*"))
            random.shuffle(images)
            n_total = len(images)
            n_train = int(train_ratio * n_total)
            n_val = int(val_ratio * n_total)

            train_files = images[:n_train]
            val_files = images[n_train:n_train+n_val]
            test_files = images[n_train+n_val:]

            for f in train_files:
                shutil.copy(f, OUTPUT_DIR / "train" / cls.name / f.name)
            for f in val_files:
                shutil.copy(f, OUTPUT_DIR / "val" / cls.name / f.name)
            for f in test_files:
                shutil.copy(f, OUTPUT_DIR / "test" / cls.name / f.name)

    print("✅ Dataset berhasil di-split!")

if __name__ == "__main__":
    split_dataset()
