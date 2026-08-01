"""
Project Configuration
"""

from pathlib import Path
import torch

# ===========================
# Paths
# ===========================

ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = ROOT / "data" / "processed" / "train_augmented"
VALIDATION_DIR = ROOT / "data" / "processed" / "validation"
TEST_DIR = ROOT / "data" / "raw" / "test"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ===========================
# Classes
# ===========================

CLASS_NAMES = [
    "Black Rot",
    "ESCA",
    "Healthy",
    "Leaf Blight"
]

NUM_CLASSES = len(CLASS_NAMES)

# ===========================
# Image
# ===========================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

NUM_WORKERS = 4

# ===========================
# Training
# ===========================

EPOCHS = 15

LEARNING_RATE = 1e-3

SEED = 42

# ===========================
# Device
# ===========================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {DEVICE}")
