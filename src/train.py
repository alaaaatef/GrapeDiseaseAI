"""
Model Training
Grape Disease Classification
"""

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config
from dataloader import get_dataloaders
from model import get_model

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ======================================================
# Train One Epoch
# ======================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(config.DEVICE, non_blocking=True)
        labels = labels.to(config.DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type="cuda"):

            outputs = model(images)

            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total

    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# ======================================================
# Validation
# ======================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(config.DEVICE, non_blocking=True)

            labels = labels.to(config.DEVICE, non_blocking=True)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

    loss = running_loss / total

    acc = correct / total

    return loss, acc


# ======================================================
# Train
# ======================================================

def train_model(model_name):

    print("="*60)

    print("Training:", model_name)

    print("="*60)

    train_loader, val_loader, _ = get_dataloaders()

    model = get_model(model_name).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(

        filter(lambda p: p.requires_grad, model.parameters()),

        lr=config.LEARNING_RATE

    )

    scheduler = ReduceLROnPlateau(

        optimizer,

        mode="min",

        factor=0.2,

        patience=3

    )

    scaler = GradScaler("cuda")

    history = {

        "train_loss": [],

        "val_loss": [],

        "train_acc": [],

        "val_acc": []

    }

    best_weights = copy.deepcopy(model.state_dict())

    best_acc = 0

    patience = 4

    wait = 0

    start = time.time()

    for epoch in range(config.EPOCHS):

        train_loss, train_acc = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler

        )

        val_loss, val_acc = validate(

            model,

            val_loader,

            criterion

        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)

        history["train_acc"].append(train_acc)

        history["val_acc"].append(val_acc)

        print(

            f"Epoch {epoch+1:02}/{config.EPOCHS}"

            f" | Train Loss {train_loss:.4f}"

            f" | Train Acc {train_acc:.4f}"

            f" | Val Loss {val_loss:.4f}"

            f" | Val Acc {val_acc:.4f}"

        )

        if val_acc > best_acc:

            best_acc = val_acc

            wait = 0

            best_weights = copy.deepcopy(model.state_dict())

            torch.save(

                {

                    "epoch": epoch + 1,

                    "model_state_dict": model.state_dict(),

                    "optimizer_state_dict": optimizer.state_dict(),

                    "best_accuracy": best_acc

                },

                config.MODEL_DIR / f"{model_name}.pth"

            )

        else:

            wait += 1

            if wait >= patience:

                print("Early Stopping")

                break

    end = time.time()

    model.load_state_dict(best_weights)

    print(f"\nBest Validation Accuracy : {best_acc:.4f}")

    print(f"Training Time : {(end-start)/60:.2f} minutes")

    torch.save(

        history,

        config.OUTPUT_DIR / f"{model_name}_history.pt"

    )


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    train_model("cnn")

    train_model("resnet50")

    train_model("efficientnetb7")

    print("\nTraining Finished.")