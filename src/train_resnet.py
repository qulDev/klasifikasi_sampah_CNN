import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from utils.dataset import create_dataloaders
import copy

# -----------------------
# Hyperparameters
# -----------------------
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 10  # early stopping patience

# -----------------------
# DataLoader
# -----------------------
train_loader, val_loader, class_names = create_dataloaders(
    data_dir="data", batch_size=BATCH_SIZE
)

# -----------------------
# Model (ResNet18)
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(class_names))
model = model.to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------
# Training Loop dengan EarlyStopping + Checkpoint
# -----------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float("inf")
epochs_no_improve = 0

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

    # --- Training ---
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss = val_loss / len(val_loader.dataset)

    print(f"Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

    # --- EarlyStopping & Checkpoint ---
    if val_loss < best_loss:
        print("✅ Validation loss improved, saving model...")
        best_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), "best_model.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("⏹ Early stopping triggered!")
            break

# Load best model
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "final_model.pth")
print("Training selesai. Model terbaik disimpan di best_model.pth 🚀")
