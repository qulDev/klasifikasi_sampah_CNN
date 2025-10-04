import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys
import os

# ==============================
# 🔧 Konfigurasi
# ==============================
MODEL_PATH = "checkpoints/resnet50_best.pth"
CLASS_NAMES_PATH = "checkpoints/class_names.txt"  # akan kita buat di bawah
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# 🔍 Load class names
# ==============================
if not os.path.exists(CLASS_NAMES_PATH):
    raise FileNotFoundError(f"File {CLASS_NAMES_PATH} tidak ditemukan. Buat dulu class_names.txt.")

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# ==============================
# 🧠 Load model
# ==============================
def load_model(model_path, class_names, device):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, len(class_names))
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model(MODEL_PATH, class_names, DEVICE)

# ==============================
# 🖼️ Transformasi gambar
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==============================
# 🚀 Fungsi prediksi
# ==============================
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File {image_path} tidak ditemukan.")
        return

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probs, 0)
        predicted_class = class_names[top_idx.item()]
        confidence = top_prob.item() * 100

    print(f"🧩 Prediksi: {predicted_class} ({confidence:.2f}%)")
    return predicted_class, confidence

# ==============================
# 📸 Main (CLI mode)
# ==============================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Gunakan: python src/predict_image.py path_ke_gambar.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image(image_path)
