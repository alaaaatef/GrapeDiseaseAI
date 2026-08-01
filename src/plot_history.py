import torch
import matplotlib.pyplot as plt
import config

models = ["cnn", "resnet50", "efficientnetb7"]

for model in models:

    history = torch.load(
        config.OUTPUT_DIR / f"{model}_history.pt"
    )

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title(f"{model.upper()} Loss")

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        config.OUTPUT_DIR / f"{model}_loss.png",
        dpi=300
    )

    plt.close()

    # Accuracy
    plt.figure(figsize=(8,5))

    plt.plot(
        epochs,
        history["train_acc"],
        label="Train Accuracy"
    )

    plt.plot(
        epochs,
        history["val_acc"],
        label="Validation Accuracy"
    )

    plt.title(f"{model.upper()} Accuracy")

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    plt.grid(True)

    plt.savefig(
        config.OUTPUT_DIR / f"{model}_accuracy.png",
        dpi=300
    )

    plt.close()

print("Done")