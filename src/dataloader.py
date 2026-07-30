"""
Data Loader
Grape Disease Classification Project

This module loads the training, validation,
and testing datasets using TensorFlow.
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf
import config

# ==================================================
# Data Loader
# ==================================================

def load_datasets():
    """
    Load Train, Validation and Test datasets.

    Returns:
        train_dataset
        validation_dataset
        test_dataset
    """

    # ----------------------------
    # Training Dataset
    # ----------------------------

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TRAIN_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="training",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE
    )

    # ----------------------------
    # Validation Dataset
    # ----------------------------

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TRAIN_DIR,
        validation_split=config.VALIDATION_SPLIT,
        subset="validation",
        seed=config.SEED,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE
    )

    # ----------------------------
    # Test Dataset
    # ----------------------------

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        config.RAW_TEST_DIR,
        image_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # ----------------------------
    # Improve Performance
    # ----------------------------

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(AUTOTUNE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset