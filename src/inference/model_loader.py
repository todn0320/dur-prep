import torch
import torch.nn as nn
from torchvision import models


def load_model(model_path: str, num_classes: int, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    checkpoint = torch.load(model_path, map_location=device)

    # 다양한 저장 형태 대응
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    else:
        # 혹시 torch.save(model, ...) 방식이면 그대로 사용
        model = checkpoint

    model.to(device)
    model.eval()

    return model, device