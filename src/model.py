"""
Model Architectures
Grape Disease Classification

1. Baseline CNN
2. ResNet50
3. EfficientNetB7
"""

import torch
import torch.nn as nn

from torchvision.models import (
    resnet50,
    efficientnet_b7,
    ResNet50_Weights,
    EfficientNet_B7_Weights
)

import config


# ==========================================================
# Baseline CNN
# ==========================================================

class BaselineCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # -----------------------------
            # Block 1
            # -----------------------------
            nn.Conv2d(3, 32, kernel_size=6, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # -----------------------------
            # Block 2
            # -----------------------------
            nn.Conv2d(32, 32, kernel_size=5, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # -----------------------------
            # Block 3
            # -----------------------------
            nn.Conv2d(32, 32, kernel_size=4, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # -----------------------------
            # Block 4
            # -----------------------------
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # -----------------------------
            # Block 5
            # -----------------------------
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # -----------------------------
            # Block 6
            # -----------------------------
            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((3, 3))

        )   # ← القوس ده كان ناقص

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Dropout(0.2),

            nn.Linear(32 * 3 * 3, 512),

            nn.ReLU(inplace=True),

            nn.Linear(512, 512),

            nn.ReLU(inplace=True),

            nn.Linear(
                512,
                config.NUM_CLASSES
            )

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================================
# ResNet50
# ==========================================================

def build_resnet50():

    model = resnet50(
        weights=ResNet50_Weights.IMAGENET1K_V2
    )

    # -----------------------------------
    # Freeze Backbone
    # -----------------------------------
    for param in model.parameters():
        param.requires_grad = False

    # -----------------------------------
    # Replace Classifier
    # -----------------------------------
    model.fc = nn.Sequential(

        nn.Linear(
            model.fc.in_features,
            512
        ),

        nn.ReLU(inplace=True),

        nn.Dropout(0.5),

        nn.Linear(
            512,
            config.NUM_CLASSES
        )

    )

    # -----------------------------------
    # Train only classifier
    # -----------------------------------
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


# ==========================================================
# EfficientNetB7
# ==========================================================

def build_efficientnet():

    model = efficientnet_b7(
        weights=EfficientNet_B7_Weights.DEFAULT
    )

    # -----------------------------------
    # Freeze Backbone
    # -----------------------------------
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features

    # -----------------------------------
    # Replace Classifier
    # -----------------------------------
    model.classifier = nn.Sequential(

        nn.Dropout(0.5),

        nn.Linear(
            in_features,
            512
        ),

        nn.ReLU(inplace=True),

        nn.Dropout(0.5),

        nn.Linear(
            512,
            config.NUM_CLASSES
        )

    )

    # -----------------------------------
    # Train only classifier
    # -----------------------------------
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ==========================================================
# Factory
# ==========================================================

def get_model(name):

    name = name.lower()

    if name == "cnn":
        return BaselineCNN()

    elif name == "resnet50":
        return build_resnet50()

    elif name == "efficientnetb7":
        return build_efficientnet()

    else:
        raise ValueError("Unknown model.")