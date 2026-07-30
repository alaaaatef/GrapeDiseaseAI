"""
Dataset Validation
Grape Disease Classification Project

This module validates the dataset before training.
"""

# ==================================================
# Imports
# ==================================================

import os

import config
from utils import (
    print_header,
    count_images,
    is_image_file
)

# ==================================================
# Validation Functions
# ==================================================

def validate_dataset():
    """
    Validate dataset structure before training.
    """

    print_header("DATASET VALIDATION")

    # ==================================================
    # Check Dataset Directories
    # ==================================================

    required_directories = [
        config.RAW_TRAIN_DIR,
        config.RAW_TEST_DIR
    ]

    print("Checking dataset directories...\n")

    for directory in required_directories:

        if not os.path.exists(directory):
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        print(f"Found: {directory}")

    # ==================================================
    # Check Classes
    # ==================================================

    print("\nChecking class folders...\n")

    for class_name in config.CLASS_NAMES:

        class_path = os.path.join(
            config.RAW_TRAIN_DIR,
            class_name
        )

        if not os.path.exists(class_path):
            raise FileNotFoundError(
                f"Missing class folder: {class_name}"
            )

        print(f"{class_name} ✓")

    # ==================================================
    # Count Images
    # ==================================================

    print("\nCounting images...\n")

    total_images = 0

    for class_name in config.CLASS_NAMES:

        class_path = os.path.join(
            config.RAW_TRAIN_DIR,
            class_name
        )

        image_count = count_images(class_path)

        total_images += image_count

        print(f"{class_name:<15}: {image_count}")

    print("-" * 35)
    print(f"Total Training Images : {total_images}")

    # ==================================================
    # Check Image Formats
    # ==================================================

    print("\nChecking image formats...\n")

    invalid_files = []

    for class_name in config.CLASS_NAMES:

        class_path = os.path.join(
            config.RAW_TRAIN_DIR,
            class_name
        )

        for file in os.listdir(class_path):

            if not is_image_file(file):

                invalid_files.append(
                    os.path.join(class_name, file)
                )

    if len(invalid_files) == 0:

        print("All files have valid image formats.")

    else:

        print(f"Found {len(invalid_files)} invalid file(s):\n")

        for file in invalid_files:
            print(file)

    print("\nDataset validation completed successfully.")