"""
Image Preprocessing
Grape Disease Classification Project
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf

# ==================================================
# Normalization Layer
# ==================================================

normalization = tf.keras.layers.Rescaling(1.0 / 255)

# ==================================================
# Preprocess Dataset
# ==================================================

def preprocess_dataset(dataset):
    """
    Normalize images only.

    Since the downloaded dataset is already augmented,
    no online data augmentation is applied.
    """

    dataset = dataset.map(
        lambda images, labels: (
            normalization(images),
            labels
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset