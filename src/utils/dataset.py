# src/utils/dataset.py
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def create_dataloaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1):
    """
    Membuat DataLoader untuk train, validation, dan test set
    dengan augmentasi pada train set.
    """

    # ⚙️ Augmentasi kuat untuk TRAINING
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # zoom in/out
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),  # miring kanan-kiri
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
        ),
        transforms.RandomAffine(
            degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # mean & std dari ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ✨ Transformasi ringan untuk VALIDASI & TEST
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 📂 Dataset utama (ImageFolder)
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    # 🧩 Split dataset ke train / val / test
    total_size = len(full_dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Untuk val & test, kita ubah transform jadi eval_transform
    val_dataset.dataset.transform = eval_transform
    test_dataset.dataset.transform = eval_transform

    # 🚀 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_names = full_dataset.classes

    print(f"📦 Dataset summary:")
    print(f" - Train: {len(train_dataset)}")
    print(f" - Val:   {len(val_dataset)}")
    print(f" - Test:  {len(test_dataset)}")
    print(f" - Classes: {class_names}")

    return train_loader, val_loader, test_loader, class_names
