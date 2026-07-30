"""
Dataset Visualization
Grape Disease Classification Project
"""

# ==================================================
# Imports
# ==================================================

import matplotlib.pyplot as plt
import tensorflow as tf

from preprocessing import normalization, data_augmentation

# ==================================================
# Display Images
# ==================================================

def visualize_dataset(dataset):
    """
    Display original, normalized,
    and augmented images.
    """

    # Get one batch
    images, labels = next(iter(dataset))

    # First image
    image = images[0]

    # Normalize
    normalized_image = image

    # Augment
    augmented_image = data_augmentation(
        tf.expand_dims(normalized_image, axis=0)
    )

    augmented_image = tf.squeeze(augmented_image)

    print("Original Min :", tf.reduce_min(image).numpy())
    print("Original Max :", tf.reduce_max(image).numpy())

    print("Normalized Min :", tf.reduce_min(normalized_image).numpy())
    print("Normalized Max :", tf.reduce_max(normalized_image).numpy())

    print("Augmented Min :", tf.reduce_min(augmented_image).numpy())
    print("Augmented Max :", tf.reduce_max(augmented_image).numpy())
    # Plot
    plt.figure(figsize=(15,5))

    # -------------------------
    # Original
    # -------------------------
    plt.subplot(1,3,1)

    plt.imshow(image.numpy())

    plt.title("Original")

    plt.axis("off")

    # -------------------------
    # Normalized
    # -------------------------
    plt.subplot(1,3,2)

    plt.imshow(normalized_image.numpy())

    plt.title("Normalized")

    plt.axis("off")

    # -------------------------
    # Augmented
    # -------------------------
    plt.subplot(1,3,3)

    plt.imshow(augmented_image.numpy())

    plt.title("Augmented")

    plt.axis("off")

    plt.show()