import random
from pathlib import Path

from PIL import Image

import torchvision.transforms as transforms


SOURCE = Path("data/processed/train")
TARGET = Path("data/processed/train_augmented")

TARGET_PER_CLASS = 3000

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.10,0.10),
        scale=(0.90,1.10)
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    )
])

TARGET.mkdir(exist_ok=True)

for class_dir in SOURCE.iterdir():

    output_dir = TARGET / class_dir.name
    output_dir.mkdir(exist_ok=True)

    images = list(class_dir.glob("*"))

    # copy originals
    for img in images:
        Image.open(img).save(output_dir/img.name)

    current = len(images)

    needed = TARGET_PER_CLASS-current

    print(class_dir.name,current,"->",TARGET_PER_CLASS)

    for i in range(needed):

        img_path=random.choice(images)

        image=Image.open(img_path).convert("RGB")

        aug=transform(image)

        aug.save(
            output_dir /
            f"aug_{i}.jpg"
        )

print("\nDone.")