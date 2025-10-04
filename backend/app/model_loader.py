# backend/app/model_loader.py

import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path: str, class_names: list, device: torch.device):
    # Load arsitektur dasar (ResNet50)
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features

    # Sesuaikan dengan arsitektur saat training (3 layer di fc)
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Linear(256, len(class_names))
    )

    # Load bobot model
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model
