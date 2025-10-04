import torch
import torch.nn as nn
from torchvision import models
import cv2
from torchvision import transforms
import numpy as np

# === Konfigurasi ===
MODEL_PATH = "checkpoints_v2/resnet50_best_v2.pth"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model dengan arsitektur yang sama seperti training ===
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, len(CLASS_NAMES))
)

# === Load bobot ===
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

# === Transformasi gambar ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# === Buka webcam ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak bisa membuka webcam.")
    exit()

print("🎥 Webcam aktif. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame.")
        break

    # Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        pred_class = CLASS_NAMES[preds.item()]
        confidence = torch.softmax(outputs, dim=1)[0][preds.item()].item() * 100

    # Tampilkan hasil
    text = f"{pred_class} ({confidence:.1f}%)"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Real-time Waste Classification", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
