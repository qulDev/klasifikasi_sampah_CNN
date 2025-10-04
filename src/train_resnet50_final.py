import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils.dataset import create_dataloaders
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# =======================================================
# 🚀 SETUP
# =======================================================
os.makedirs("checkpoints", exist_ok=True)

# 🔧 Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 10  # early stopping patience

# 🧠 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔍 Device: {device}")

# =======================================================
# 📦 LOAD DATA
# =======================================================
train_loader, val_loader, test_loader, class_names = create_dataloaders(
    "dataset/merged", batch_size=BATCH_SIZE
)
print(f"📂 Jumlah kelas: {len(class_names)} → {class_names}")

# =======================================================
# 🏗️ MODEL SETUP (ResNet50)
# =======================================================
model = models.resnet50(weights="IMAGENET1K_V1")

# Freeze sebagian layer agar training lebih cepat dan stabil
for param in model.layer1.parameters():
    param.requires_grad = False
for param in model.layer2.parameters():
    param.requires_grad = False

# Ganti FC layer (harus sama nanti saat inference)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(num_ftrs, len(class_names))
)
model = model.to(device)

# =======================================================
# ⚙️ LOSS, OPTIMIZER, DAN SCHEDULER
# =======================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# =======================================================
# 🏋️ TRAINING LOOP
# =======================================================
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_loss = float("inf")
epochs_no_improve = 0
best_model_wts = copy.deepcopy(model.state_dict())

log_file = open("checkpoints/training_log.txt", "w")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    log_file.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS}\n")

    # --- TRAINING ---
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

    # --- VALIDATION ---
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

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    scheduler.step(val_loss)

    # --- LOGGING ---
    train_losses.append(epoch_loss)
    val_losses.append(val_loss)
    train_accs.append(epoch_acc.item())
    val_accs.append(val_acc.item())

    log_line = f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"
    print(log_line.strip())
    log_file.write(log_line)

    # --- EARLY STOPPING ---
    if val_loss < best_loss:
        print("✅ Validation loss improved, saving model...")
        log_file.write("✅ Validation loss improved, saving model...\n")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "checkpoints/resnet50_best.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered!")
            log_file.write("⏹ Early stopping triggered!\n")
            break

# =======================================================
# 🔥 SAVE MODEL
# =======================================================
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "checkpoints/resnet50_final.pth")
print("🎉 Training selesai! Model terbaik disimpan di checkpoints/resnet50_best.pth")
log_file.write("🎉 Training selesai! Model terbaik disimpan di checkpoints/resnet50_best.pth\n")

# =======================================================
# 📈 PLOTTING HASIL TRAINING
# =======================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("checkpoints/training_plot.png")
print("📊 Grafik training disimpan di checkpoints/training_plot.png")

# =======================================================
# 🧪 TEST SET EVALUATION
# =======================================================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- REPORT ---
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print("\n📑 Classification Report:\n")
print(report)

with open("checkpoints/classification_report.txt", "w") as f:
    f.write(report)

# --- CONFUSION MATRIX ---
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("checkpoints/confusion_matrix.png")
print("📊 Confusion matrix disimpan di checkpoints/confusion_matrix.png")

log_file.close()
