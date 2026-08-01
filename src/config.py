"""
Project Configuration
Grape Disease Classification Project
"""

# ==================================================
# Project Information
# ==================================================
PROJECT_NAME = "Grape Disease Classification"
PROJECT_VERSION = "1.0.0"

# ==================================================
# Random Seed
# ==================================================
SEED = 42

# ==================================================
# Image Settings
# ==================================================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# ==================================================
# Dataset Split
# ==================================================
VALIDATION_SPLIT = 0.2

# ==================================================
# Dataset Paths
# ==================================================
RAW_TRAIN_DIR = "data/raw/train"
RAW_TEST_DIR = "data/raw/test"

PROCESSED_TRAIN_DIR = "data/processed/train"
PROCESSED_VAL_DIR = "data/processed/validation"
PROCESSED_TEST_DIR = "data/processed/test"

REPORTS_DIR = "data/reports"

# ==================================================
# Class Information
# ==================================================
CLASS_NAMES = [
    "Black Rot",
    "Esca",
    "Healthy",
    "Leaf Blight"
]

NUM_CLASSES = len(CLASS_NAMES)

# ==================================================
# Supported Image Formats
# ==================================================
SUPPORTED_FORMATS = (
    ".jpg",
    ".jpeg",
    ".png"
)

# ==================================================
# Training Settings
# ==================================================
SHUFFLE = True


# ==================================================
# Training
# ==================================================

EPOCHS = 20

# ==================================================
# Save Paths
# ==================================================

MODEL_SAVE_PATH = "models"

HISTORY_SAVE_PATH = "history"
