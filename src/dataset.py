"""
Dataset Loader
Grape Disease Classification
PyTorch Version
"""

# ==========================================
# Imports
# ==========================================

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

import config

# ==========================================
# Training Transform
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10)
    ),

    transforms.RandomResizedCrop(
        config.IMAGE_SIZE,
        scale=(0.9, 1.0)
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ==========================================
# Validation Transform
# ==========================================

validation_transform = transforms.Compose([

    transforms.Resize(
        (config.IMAGE_SIZE, config.IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])

# ==========================================
# Test Transform
# ==========================================

test_transform = validation_transform

# ==========================================
# Dataset
# ==========================================

train_dataset = datasets.ImageFolder(

    root=config.TRAIN_DIR,

    transform=train_transform

)

validation_dataset = datasets.ImageFolder(

    root=config.VAL_DIR,

    transform=validation_transform

)

test_dataset = datasets.ImageFolder(

    root=config.TEST_DIR,

    transform=test_transform

)

# ==========================================
# DataLoader
# ==========================================

train_loader = DataLoader(

    train_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=True,

    num_workers=config.NUM_WORKERS,

    pin_memory=True

)

validation_loader = DataLoader(

    validation_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False,

    num_workers=config.NUM_WORKERS,

    pin_memory=True

)

test_loader = DataLoader(

    test_dataset,

    batch_size=config.BATCH_SIZE,

    shuffle=False,

    num_workers=config.NUM_WORKERS,

    pin_memory=True

)

# ==========================================
# Test
# ==========================================

if __name__ == "__main__":

    print("Train :", len(train_dataset))
    print("Validation :", len(validation_dataset))
    print("Test :", len(test_dataset))

    print(train_dataset.classes)