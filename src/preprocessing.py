"""
Image Preprocessing
Grape Disease Classification Project
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf

# ==================================================
# Data Augmentation
# ==================================================

data_augmentation = tf.keras.Sequential([

    tf.keras.layers.RandomFlip(
        "horizontal"
    ),

    tf.keras.layers.RandomRotation(
        0.10
    ),

    tf.keras.layers.RandomZoom(
        0.10
    ),

    tf.keras.layers.RandomContrast(
        0.10
    )

])

# ==================================================
# Normalization Layer
# ==================================================

normalization = tf.keras.layers.Rescaling(
    1.0 / 255
)

# ==================================================
# Preprocess Dataset
# ==================================================

def preprocess_dataset(dataset, training=True):

    """
    Apply preprocessing to dataset.

    Parameters
    ----------
    dataset : tf.data.Dataset

    training : bool

    Returns
    -------
    tf.data.Dataset
    """

    # Normalize Images

    dataset = dataset.map(

        lambda images, labels: (

            normalization(images),

            labels

        ),

        num_parallel_calls=tf.data.AUTOTUNE

    )

    # Apply Data Augmentation only for Training

    if training:

        dataset = dataset.map(

            lambda images, labels: (

                data_augmentation(images),

                labels

            ),

            num_parallel_calls=tf.data.AUTOTUNE

        )

    # Improve Performance

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset