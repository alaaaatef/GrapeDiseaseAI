"""
Data Loader
Grape Disease Classification Project

This module loads the training,
validation and testing datasets.
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf
import config

# ==================================================
# Load Datasets
# ==================================================

def load_datasets():
    """
    Load train, validation and test datasets.
    """

    # -----------------------------
    # Train Dataset
    # -----------------------------
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TRAIN_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE
    )

    # -----------------------------
    # Validation Dataset
    # -----------------------------
    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TRAIN_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Test Dataset
    # -----------------------------
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TEST_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # -----------------------------
    # Improve Performance
    # -----------------------------
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


# ==================================================
# Test Loader
# ==================================================

if __name__ == "__main__":

    train_dataset, validation_dataset, test_dataset = load_datasets()

    print("\nDatasets Loaded Successfully!\n")

    print("Train batches      :", train_dataset.cardinality().numpy())
    print("Validation batches :", validation_dataset.cardinality().numpy())
    print("Test batches       :", test_dataset.cardinality().numpy())