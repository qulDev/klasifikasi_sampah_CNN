import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np

# --- Konfigurasi ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/resnet50_best.pth"

# Kelas sesuai dataset kamu
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# --- Load model ---
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- Transformasi gambar ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Buka Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak dapat membuka kamera!")
    exit()

print("📷 Tekan 'q' untuk keluar dari realtime detection")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Salin frame untuk prediksi
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    # Prediksi
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        label = CLASSES[preds.item()]

    # --- Tampilkan di layar ---
    cv2.putText(frame, f"{label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Realtime Sampah Classifier", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
