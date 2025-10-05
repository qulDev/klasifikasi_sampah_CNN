import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

# ======================================
# 🧠 Konfigurasi
# ======================================
MODEL_PATH = "checkpoints_v2/resnet50_best_v2.pth"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================
# 🏗️ Load model ResNet50
# ======================================
def load_resnet50(model_path, class_names, device):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),  # fc.0
        nn.ReLU(),  # fc.1
        nn.Linear(256, len(class_names))  # fc.2
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ======================================
# 🔄 Preprocessing gambar
# ======================================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ======================================
# 🎥 Realtime Webcam Prediction
# ======================================
def main():
    model = load_resnet50(MODEL_PATH, CLASS_NAMES, DEVICE)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Tidak dapat membuka kamera!")
        return

    print("🎥 Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- Preprocess ---
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        # --- Prediksi ---
        with torch.no_grad():
            output = model(tensor)
            pred_idx = output.argmax(1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = torch.softmax(output, dim=1)[0][pred_idx].item()

        # --- Tampilkan hasil ---
        label_text = f"{pred_label.upper()} ({confidence*100:.1f}%)"
        cv2.putText(frame, label_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Realtime Waste Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
