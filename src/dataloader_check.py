import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# lokasi dataset hasil split
DATA_DIR = "dataset/split"

# augmentasi untuk training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# hanya resize + normalisasi untuk val/test
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# load dataset
train_dataset = datasets.ImageFolder(root=f"{DATA_DIR}/train", transform=train_transform)
val_dataset   = datasets.ImageFolder(root=f"{DATA_DIR}/val", transform=test_transform)
test_dataset  = datasets.ImageFolder(root=f"{DATA_DIR}/test", transform=test_transform)

# buat DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

# cek apakah CUDA tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔍 Device:", device)
print("📊 Jumlah batch training:", len(train_loader))
print("📊 Jumlah batch validation:", len(val_loader))
print("📊 Jumlah batch test:", len(test_loader))

# ambil 1 batch sample
images, labels = next(iter(train_loader))
print("🖼️ Batch shape:", images.shape)
print("🏷️ Label shape:", labels.shape)
