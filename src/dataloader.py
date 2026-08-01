"""
PyTorch Data Loader
Grape Disease Classification Project
"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


# ======================================================
# Dataset
# ======================================================

class GrapeDataset(Dataset):

    def __init__(self, root_dir, transform=None):

        self.transform = transform
        self.image_paths = []
        self.labels = []

        self.class_names = sorted(
            [
                folder
                for folder in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, folder))
            ]
        )

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.class_names)
        }

        for cls in self.class_names:

            folder = os.path.join(root_dir, cls)

            for image_name in os.listdir(folder):

                if image_name.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):

                    self.image_paths.append(
                        os.path.join(folder, image_name)
                    )

                    self.labels.append(
                        self.class_to_idx[cls]
                    )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        image = Image.open(
            self.image_paths[index]
        ).convert("RGB")

        label = self.labels[index]

        if self.transform:

            image = self.transform(image)

        return image, label


# ======================================================
# Train Transform
# ======================================================

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# ======================================================
# Validation / Test Transform
# ======================================================

validation_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# ======================================================
# DataLoaders
# ======================================================

def get_dataloaders():

    train_dataset = GrapeDataset(

        config.TRAIN_DIR,

        transform=train_transform

    )

    validation_dataset = GrapeDataset(

        config.VALIDATION_DIR,

        transform=validation_transform

    )

    test_dataset = GrapeDataset(

        config.TEST_DIR,

        transform=validation_transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=True,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=True

    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=True

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=config.BATCH_SIZE,

        shuffle=False,

        num_workers=config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        persistent_workers=True

    )

    return train_loader, validation_loader, test_loader


# ======================================================
# Test
# ======================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = get_dataloaders()

    print()

    print(f"Device: {config.DEVICE}")

    print()

    print(f"Train Images : {len(train_loader.dataset)}")

    print(f"Validation   : {len(val_loader.dataset)}")

    print(f"Test         : {len(test_loader.dataset)}")

    print()

    print("Classes")

    print(train_loader.dataset.class_names)

    print()

    x, y = next(iter(train_loader))

    print("Batch Shape :", x.shape)

    print("Labels Shape:", y.shape)