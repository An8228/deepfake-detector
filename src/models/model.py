import torch.nn as nn
from torchvision import models


def build_model(cfg, pretrained=True):
    if cfg['model_name'] != 'efficientnet_b0':
        raise ValueError(f"Unsupported model: {cfg['model_name']}")
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)  # single logit: >0 means fake
    return model