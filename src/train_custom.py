# src/train_custom.py
import os
import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from tqdm import tqdm

from model.custom_cnn import CustomCNN

# Paths
DATA_DIR = Path("dataset/split")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Hyperparams
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4  # sesuaikan dengan CPU kamu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# transforms
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# datasets & loaders
train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_transform)
val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=val_transform)
test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

class_names = train_ds.classes
num_classes = len(class_names)
print("Classes:", class_names)

# compute class weights from training data to handle imbalance
def compute_class_weights(dataset):
    counts = [0] * num_classes
    for _, label in dataset:
        counts[label] += 1
    counts = np.array(counts)
    # inverse frequency
    weights = counts.sum() / (counts + 1e-9)
    weights = weights / weights.sum() * num_classes  # normalize -> mean around 1
    return torch.tensor(weights, dtype=torch.float32)

class_weights = compute_class_weights(train_ds).to(device)
print("Class weights:", class_weights.cpu().numpy())

# model, loss, optimizer, scheduler
model = CustomCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

# training + validation functions
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{correct/total:.4f}"})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# training loop
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\n===== Epoch {epoch}/{NUM_EPOCHS} =====")
    t0 = time.time()
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_labels, val_preds = validate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    scheduler.step(val_acc)

    print(f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")
    print(f"Epoch time: {time.time()-t0:.1f}s")

    # simpan model terbaik berdasar val_acc
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, CHECKPOINT_DIR / "best_checkpoint.pth")
        print("Saved best model checkpoint.")

# load best model weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
print("Training selesai. Best val acc:", best_acc)

# akhir: evaluasi di test set
print("\nEvaluating on test set...")
test_loss, test_acc, test_labels, test_preds = validate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}")

print("\nClassification Report:")
print(classification_report(test_labels, test_preds, target_names=class_names, digits=4))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print(cm)
