import cv2
import torch
import numpy as np
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO

# =============================
# Konfigurasi
# =============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints_v2/resnet50_best_v2.pth"

# Nama kelas sesuai training kamu
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# =============================
# Load ResNet50 kamu
# =============================
def load_resnet50(model_path, class_names, device):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),  # fc.0
        nn.ReLU(),                 # fc.1
        nn.Linear(256, len(class_names))  # fc.2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# =============================
# Load YOLOv8 (pretrained COCO)
# =============================
yolo_model = YOLO("yolov8n.pt")  # versi kecil dan cepat
resnet_model = load_resnet50(MODEL_PATH, CLASS_NAMES, DEVICE)

# =============================
# Transformasi untuk ResNet50
# =============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# =============================
# Realtime Detection
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Tidak bisa membuka kamera")
    exit()

print("✅ Kamera aktif, tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek pakai YOLO
    results = yolo_model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = yolo_model.names[cls_id]
        conf = float(box.conf[0])

        # Ambil koordinat bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop objek yang terdeteksi
        crop = frame[y1:y2, x1:x2]

        # Hanya klasifikasi jika objek cukup besar
        if crop.size > 0 and crop.shape[0] > 50 and crop.shape[1] > 50:
            try:
                img = transform(crop).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    outputs = resnet_model(img)
                    _, pred = torch.max(outputs, 1)
                    class_name = CLASS_NAMES[pred.item()]
            except Exception as e:
                class_name = "Unknown"
        else:
            class_name = "Unknown"

        # Gambar bounding box + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} ({conf:.2f}) → {class_name}"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Tampilkan hasil
    cv2.imshow("YOLO + ResNet50 Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
