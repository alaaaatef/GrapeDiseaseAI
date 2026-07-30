"""
Utility Functions
Grape Disease Classification Project

This module contains reusable utility functions
used across the project.
"""

# ==================================================
# Imports
# ==================================================

import os
import config


# ==================================================
# Utility Functions
# ==================================================

def create_directory(path):
    """
    Create a directory if it does not exist.

    Args:
        path (str): Directory path.
    """

    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory created: {path}")


def is_image_file(filename):
    """
    Check whether a file is a supported image.

    Args:
        filename (str): File name.

    Returns:
        bool
    """

    return filename.lower().endswith(config.SUPPORTED_FORMATS)


def count_images(directory):
    """
    Count the number of image files inside a directory
    and all its subfolders.

    Args:
        directory (str): Dataset directory.

    Returns:
        int: Number of images.
    """

    total_images = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                total_images += 1

    return total_images


def print_header(title):
    """
    Print a formatted header.

    Args:
        title (str): Header title.
    """

    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)
