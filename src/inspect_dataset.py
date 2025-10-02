from pathlib import Path

def inspect_dataset():
    DATASET_PATH = Path("dataset/merged")

    print(f"Menginspeksi: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        print("❌ Folder dataset tidak ditemukan.")
        return

    total = 0
    print("Jumlah file per kelas:")
    for class_dir in sorted(DATASET_PATH.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            total += count
            print(f"  {class_dir.name:<10}: {count}")
    print(f"\nTotal gambar: {total}")

if __name__ == "__main__":
    inspect_dataset()
