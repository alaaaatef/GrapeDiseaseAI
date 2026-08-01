"""
Model Training
Grape Disease Classification Project

This module trains:
1. CNN
2. ResNet50
3. EfficientNetB7
"""

# ==================================================
# Imports
# ==================================================

import os
import pickle
import tensorflow as tf

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

import config
from dataloader import load_datasets
from model import (
    build_cnn,
    build_resnet50,
    build_efficientnetb7
)

# ==================================================
# Create Directories
# ==================================================

os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(config.HISTORY_SAVE_PATH, exist_ok=True)

# ==================================================
# Train Function
# ==================================================

def train_model(model, model_name):

    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}\n")

    train_dataset, validation_dataset, _ = load_datasets()

    checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            config.MODEL_SAVE_PATH,
            f"{model_name}.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        verbose=1
    )

    history = model.fit(

        train_dataset,

        validation_data=validation_dataset,

        epochs=config.EPOCHS,

        callbacks=[
            checkpoint,
            early_stop,
            reduce_lr
        ]

    )

    # ============================================
    # Save History
    # ============================================

    history_path = os.path.join(
        config.HISTORY_SAVE_PATH,
        f"{model_name}_history.pkl"
    )

    with open(history_path, "wb") as file:
        pickle.dump(history.history, file)

    print(f"\nHistory saved to: {history_path}")

    return history


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    # ============================================
    # CNN
    # ============================================

    cnn_model = build_cnn()
    train_model(cnn_model, "cnn")

    # ============================================
    # ResNet50
    # ============================================

    resnet_model = build_resnet50()
    train_model(resnet_model, "resnet50")

    # ============================================
    # EfficientNetB7
    # ============================================

    efficientnet_model = build_efficientnetb7()
    train_model(efficientnet_model, "efficientnetb7")

    print("\nAll models trained successfully!")