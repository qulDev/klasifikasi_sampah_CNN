from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from backend.app.model_loader import load_model
from backend.app.predict import predict_image


# === Inisialisasi FastAPI ===
app = FastAPI(title="Klasifikasi Sampah API", version="1.0")

# === Load class names ===
with open("backend/app/class_names.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

# === Setup model & device ===
MODEL_PATH = "backend/checkpoints/resnet50_best.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, CLASS_NAMES, device)

# === Endpoint utama ===
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        predicted_class = predict_image(model, CLASS_NAMES, contents, device)
        return JSONResponse(content={"class": predicted_class})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
