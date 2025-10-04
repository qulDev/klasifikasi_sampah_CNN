# src/train_resnet50_v2.py
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from utils.dataset import create_dataloaders

# 📂 Pastikan folder checkpoints ada
os.makedirs("checkpoints_v2", exist_ok=True)

# 🔧 Hyperparameters
NUM_EPOCHS = 60
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 10
WEIGHT_DECAY = 1e-4

# 📊 Dataset
train_loader, val_loader, test_loader, class_names = create_dataloaders(
    "dataset/merged", batch_size=BATCH_SIZE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Device: {device}")

# 🏗️ Load ResNet50 pretrained
model = models.resnet50(weights="IMAGENET1K_V1")

# 🔒 Freeze beberapa layer bawah biar fokus di feature tinggi
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# 🧠 Ganti FC layer sesuai versi baru
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_names))
)
model = model.to(device)

# ⚙️ Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 📈 Logging
train_losses, val_losses = [], []
train_accs, val_accs = [], []
log_file = open("checkpoints_v2/training_log.txt", "w")

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float("inf")
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    log_file.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS}\n")
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    # --- Training ---
    model.train()
    running_loss, running_corrects = 0.0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    scheduler.step(val_loss)

    # 📊 Logging
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    train_accs.append(epoch_acc.item())
    val_accs.append(val_acc.item())

    log_line = f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
    print(log_line.strip())
    log_file.write(log_line)

    # --- EarlyStopping ---
    if val_loss < best_loss:
        print("✅ Validation loss improved, saving model...")
        log_file.write("✅ Validation loss improved, saving model...\n")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "checkpoints_v2/resnet50_best_v2.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered!")
            log_file.write("⏹ Early stopping triggered!\n")
            break

# 🔥 Load best model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "checkpoints_v2/resnet50_final_v2.pth")
print("🎉 Training selesai! Model terbaik disimpan di checkpoints_v2/resnet50_best_v2.pth 🚀")
log_file.write("🎉 Training selesai! Model terbaik disimpan di checkpoints_v2/resnet50_best_v2.pth 🚀\n")

# 📈 Plot grafik
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
plt.title("Training vs Validation Loss")

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
plt.title("Training vs Validation Accuracy")

plt.tight_layout()
plt.savefig("checkpoints_v2/training_plot_v2.png")
print("📊 Grafik training disimpan di checkpoints_v2/training_plot_v2.png")

# 🔍 Evaluasi
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 📑 Report
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\n📑 Classification Report:\n")
print(report)
with open("checkpoints_v2/classification_report_v2.txt", "w") as f:
    f.write(report)

# 📊 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("checkpoints_v2/confusion_matrix_v2.png")
print("📊 Confusion matrix disimpan di checkpoints_v2/confusion_matrix_v2.png")

log_file.close()
