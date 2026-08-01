"""
Balance Training Dataset

This script creates a balanced dataset
with 1656 images per class.
"""

import os
import random
import shutil

import config

# ======================================

TARGET_IMAGES = 1656

random.seed(config.SEED)

# ======================================

SOURCE = config.RAW_TRAIN_DIR
DESTINATION = config.PROCESSED_TRAIN_DIR

os.makedirs(DESTINATION, exist_ok=True)

# ======================================

for class_name in os.listdir(SOURCE):

    source_class = os.path.join(SOURCE, class_name)

    if not os.path.isdir(source_class):
        continue

    destination_class = os.path.join(
        DESTINATION,
        class_name
    )

    os.makedirs(destination_class, exist_ok=True)

    images = [

        img for img in os.listdir(source_class)

        if img.lower().endswith(
            config.SUPPORTED_FORMATS
        )

    ]

    random.shuffle(images)

    selected = images[:TARGET_IMAGES]

    for img in selected:

        shutil.copy2(

            os.path.join(source_class, img),

            os.path.join(destination_class, img)

        )

    print(

        f"{class_name} -> {len(selected)} images"

    )

print("\nBalanced dataset created successfully.")