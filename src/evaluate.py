"""
Model Evaluation
Grape Disease Classification
"""

import time
import matplotlib.pyplot as plt

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

import config
from dataloader import get_dataloaders
from model import get_model


# ==========================================================
# Evaluate
# ==========================================================

def evaluate(model_name):

    print("="*60)
    print(f"Evaluating {model_name}")
    print("="*60)

    _, _, test_loader = get_dataloaders()

    model = get_model(model_name)

    checkpoint = torch.load(
        config.MODEL_DIR / f"{model_name}.pth",
        map_location=config.DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(config.DEVICE)

    model.eval()

    y_true = []
    y_pred = []

    total_time = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                config.DEVICE,
                non_blocking=True
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()

            total_time += end - start

            predictions = outputs.argmax(1)

            y_true.extend(labels.numpy())

            y_pred.extend(predictions.cpu().numpy())

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

    print(classification_report(

        y_true,

        y_pred,

        target_names=config.CLASS_NAMES

    ))

    cm = confusion_matrix(

        y_true,

        y_pred

    )

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

        f"{model_name}_confusion_matrix.png",

        dpi=300

    )

    plt.show()
    plt.close()

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

    evaluate("cnn")

    evaluate("resnet50")

    evaluate("efficientnetb7")