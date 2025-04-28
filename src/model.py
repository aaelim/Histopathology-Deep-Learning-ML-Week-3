"""Model factory.

Keeps all model-related code in one place so training / inference
scripts can call `get_model()` without duplicating architecture logic.
"""
import torch
from torch import nn
from torchvision import models

def get_model():
    """Return a ResNet-18 fine-tuned for binary classification.

    * Why ResNet-18? – lightweight (~11 M params) yet strong baseline.
    * Why replace `fc` with a single-unit head? – we need one logit
      for BCEWithLogitsLoss (|classes| = 1).
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)  # 1 logit → sigmoid → prob
    return model
