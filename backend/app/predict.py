import torch
from torchvision import transforms
from PIL import Image
import io

def get_transform():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def predict_image(model, class_names, file_bytes, device):
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    return predicted_class
