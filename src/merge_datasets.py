import os
import shutil
from pathlib import Path

# Lokasi dataset asal
TRASHNET_DIR = Path("dataset/trashnet/data/dataset-resized/dataset-resized")
KAGGLE_DIR = Path("dataset/kaggle/garbage_classification")

# Lokasi dataset gabungan
MERGED_DIR = Path("dataset/merged")

# Mapping class
CLASS_MAPPING = {
    "cardboard": ["cardboard"],
    "glass": ["glass", "brown-glass", "green-glass", "white-glass"],
    "metal": ["metal"],
    "paper": ["paper"],
    "plastic": ["plastic"],
    "trash": ["trash", "clothes", "shoes", "battery", "biological"],
}


def prepare_folders():
    """Buat folder target jika belum ada"""
    for cls in CLASS_MAPPING.keys():
        out_dir = MERGED_DIR / cls
        out_dir.mkdir(parents=True, exist_ok=True)


def copy_files(src_dir, src_classes):
    """Salin file dari kelas sumber ke kelas target"""
    for target_class, source_classes in CLASS_MAPPING.items():
        for src_class in source_classes:
            src_path = src_dir / src_class
            if not src_path.exists():
                continue
            for file in src_path.glob("*.*"):
                # nama file unik biar tidak bentrok
                dst_file = MERGED_DIR / target_class / f"{src_class}_{file.name}"
                shutil.copy(file, dst_file)


def main():
    prepare_folders()

    print("Menggabungkan dari TrashNet...")
    copy_files(TRASHNET_DIR, CLASS_MAPPING)

    print("Menggabungkan dari Kaggle Garbage Classification...")
    copy_files(KAGGLE_DIR, CLASS_MAPPING)

    print("Selesai ✅ Dataset tersimpan di:", MERGED_DIR)


if __name__ == "__main__":
    main()
