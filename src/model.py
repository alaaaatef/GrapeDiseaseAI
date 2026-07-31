"""
Model Architectures
Grape Disease Classification Project
"""

# ==================================================
# Imports
# ==================================================

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB7

import config


# ==================================================
# Build CNN Model
# ==================================================

def build_cnn():
    """
    Build Baseline CNN Model
    """

    model = Sequential()

    # ==================================================
    # Input Layer
    # ==================================================

    model.add(
        layers.Input(
            shape=(
                config.IMAGE_SIZE[0],
                config.IMAGE_SIZE[1],
                3
            )
        )
    )

    # ==================================================
    # CNN Block 1
    # ==================================================

    model.add(
        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # ==================================================
    # CNN Block 2
    # ==================================================

    model.add(
        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # ==================================================
    # CNN Block 3
    # ==================================================

    model.add(
        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same"
        )
    )

    model.add(
        layers.BatchNormalization()
    )

    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # ==================================================
    # Flatten
    # ==================================================

    model.add(
        layers.Flatten()
    )

    # ==================================================
    # Dense Layer
    # ==================================================

    model.add(
        layers.Dense(
            256,
            activation="relu"
        )
    )

    # ==================================================
    # Dropout
    # ==================================================

    model.add(
        layers.Dropout(
            0.5
        )
    )

    # ==================================================
    # Output Layer
    # ==================================================

    model.add(
        layers.Dense(
            config.NUM_CLASSES,
            activation="softmax"
        )
    )

    # ==================================================
    # Compile Model
    # ==================================================

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nCNN Model Summary:\n")
    model.summary()

    return model


# ==================================================
# Test Model
# ==================================================

if __name__ == "__main__":
    build_cnn()




# ==================================================
# Build ResNet50 Model
# ==================================================

def build_resnet50():

    # Load Pretrained ResNet50
    base_model = ResNet50(

        weights="imagenet",

        include_top=False,

        input_shape=(

            config.IMAGE_SIZE[0],
            config.IMAGE_SIZE[1],
            3

        )

    )

    # Freeze pretrained layers
    base_model.trainable = False

    model = Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(

            256,

            activation="relu"

        ),

        layers.Dropout(

            0.5

        ),

        layers.Dense(

            config.NUM_CLASSES,

            activation="softmax"

        )

    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("\nResNet50 Model Summary:\n")

    model.summary()

    return model

# ==================================================
# Test Model
# ==================================================

if __name__ == "__main__":
        build_resnet50()



# ==================================================
# Build EfficientNetB7 Model
# ==================================================

def build_efficientnetb7():

    # Load pretrained EfficientNetB7

    base_model = EfficientNetB7(

        weights="imagenet",

        include_top=False,

        input_shape=(

            config.IMAGE_SIZE[0],
            config.IMAGE_SIZE[1],
            3

        )

    )

    # Freeze pretrained layers

    base_model.trainable = False

    model = Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(

            256,

            activation="relu"

        ),

        layers.Dropout(

            0.5

        ),

        layers.Dense(

            config.NUM_CLASSES,

            activation="softmax"

        )

    ])

    model.compile(

        optimizer="adam",

        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]

    )

    print("\nEfficientNetB7 Model Summary:\n")

    model.summary()

    return model

# ==================================================
# Test Model
# ==================================================

if __name__ == "__main__":
    build_efficientnetb7()