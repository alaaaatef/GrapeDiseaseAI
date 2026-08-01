"""
Model Evaluation
Grape Disease Classification Project

This module evaluates:
1. CNN
2. ResNet50
3. EfficientNetB7
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import config
from data_loader import load_datasets

# ==================================================
# Evaluate Function
# ==================================================

def evaluate_model(model_name):

    print(f"\n========== Evaluating {model_name} ==========\n")

    # Load model
    model = tf.keras.models.load_model(
        f"{config.MODEL_SAVE_PATH}/{model_name}.keras"
    )

    # Load Test Dataset
    _, _, test_dataset = load_datasets()

    # Predictions
    y_true = []
    y_pred = []

    for images, labels in test_dataset:

        predictions = model.predict(images, verbose=0)

        predicted_labels = np.argmax(predictions, axis=1)

        y_true.extend(labels.numpy())

        y_pred.extend(predicted_labels)

    # ==================================================
    # Metrics
    # ==================================================

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # ==================================================
    # Classification Report
    # ==================================================

    print("\nClassification Report\n")

    print(

        classification_report(

            y_true,
            y_pred,
            target_names=config.CLASS_NAMES

        )

    )

    # ==================================================
    # Confusion Matrix
    # ==================================================

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8,6))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=config.CLASS_NAMES,

        yticklabels=config.CLASS_NAMES

    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.show()


# ==================================================
# Main
# ==================================================

if __name__ == "__main__":

    evaluate_model("cnn")

    evaluate_model("resnet50")

    evaluate_model("efficientnetb7")