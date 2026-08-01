"""
Max Voting Ensemble
Grape Disease Classification
"""

import time
from collections import Counter

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

import config
from dataloader import get_dataloaders
from model import get_model


# ==========================================================
# Load Models
# ==========================================================

def load_models():

    models = []

    model_names = [
        "cnn",
        "resnet50",
        "efficientnetb7"
    ]

    for name in model_names:

        model = get_model(name)

        checkpoint = torch.load(

            config.MODEL_DIR / f"{model_name}.pth",

            map_location=config.DEVICE

        )

        model.load_state_dict(

            checkpoint["model_state_dict"]

        )

        model.to(config.DEVICE)

        model.eval()

        models.append(model)

    return models


# ==========================================================
# Majority Voting
# ==========================================================

def majority_vote(predictions):

    votes = Counter(predictions)

    highest = votes.most_common()

    if len(highest) == 1:
        return highest[0][0]

    if highest[0][1] > highest[1][1]:
        return highest[0][0]

    # Tie
    return predictions[-1]


# ==========================================================
# Ensemble Evaluation
# ==========================================================

def evaluate_ensemble():

    print("=" * 60)
    print("Evaluating Max Voting Ensemble")
    print("=" * 60)

    _, _, test_loader = get_dataloaders()

    models = load_models()

    y_true = []
    y_pred = []

    total_time = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(config.DEVICE)

            start = time.perf_counter()

            outputs = []

            for model in models:

                prediction = model(images).argmax(1)

                outputs.append(prediction.cpu().numpy())

            end = time.perf_counter()

            total_time += end - start

            outputs = np.stack(outputs)

            for i in range(images.size(0)):

                preds = outputs[:, i]

                final_prediction = majority_vote(preds)

                y_pred.append(final_prediction)

            y_true.extend(labels.numpy())

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

    print()

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()

    print(

        classification_report(

            y_true,
            y_pred,
            target_names=config.CLASS_NAMES

        )

    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=config.CLASS_NAMES

    )

    disp.plot(
        cmap="Blues",
        values_format="d"
    )

    plt.tight_layout()

    plt.savefig(

        config.OUTPUT_DIR /

        "ensemble_confusion_matrix.png",

        dpi=300

    )

    plt.show()

    inference_time = (

        total_time /

        len(test_loader.dataset)

    ) * 1000

    print()

    print(f"Average Inference Time : {inference_time:.2f} ms/image")


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    evaluate_ensemble()