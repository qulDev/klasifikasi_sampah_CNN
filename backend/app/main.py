from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
from .model_loader import load_model
from .predict import predict_image

# Inisialisasi FastAPI
app = FastAPI(title="Waste Classification API", version="1.0")

# Allow frontend (Next.js) untuk akses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model dan class names
with open("app/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

model = load_model("checkpoints/resnet50_best.pth", len(class_names))

@app.get("/")
def root():
    return {"message": "Waste Classification API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    pred_class = predict_image(model, image, class_names)
    return {"prediction": pred_class}
