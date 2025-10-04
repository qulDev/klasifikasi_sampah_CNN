import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path: str, num_classes: int):
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model
