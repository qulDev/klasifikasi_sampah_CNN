import torch
from PIL import Image
from .utils import preprocess_image

def predict_image(model, image: Image.Image, class_names: list):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, preds = torch.max(outputs, 1)
    return class_names[preds.item()]
