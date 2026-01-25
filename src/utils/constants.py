"""
Project-wide constants and paths.

This module provides centralized access to all project constants,
avoiding magic numbers and strings throughout the codebase.
"""

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Dataset paths
BRISC_DIR = DATA_DIR / "brisc2025"
CLASSIFICATION_DATA_DIR = BRISC_DIR / "classification_task"
SEGMENTATION_DATA_DIR = BRISC_DIR / "segmentation_task"

# ═══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Tumor type abbreviations (from filename parsing)
TUMOR_ABBREV = {
    "gl": "glioma",
    "me": "meningioma",
    "nt": "no_tumor",
    "pi": "pituitary",
}

# Anatomical planes
PLANES = ["ax", "co", "sa"]  # axial, coronal, sagittal
PLANE_NAMES = {
    "ax": "axial",
    "co": "coronal",
    "sa": "sagittal",
}

# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

IMG_SIZE = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE = IMG_SIZE + (IMG_CHANNELS,)

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 30
DEFAULT_VAL_SPLIT = 0.2
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL NAMES
# ═══════════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = [
    "ResNet152V2",
    "DenseNet201",
    "VGG16",
    "EfficientNetV2S",
    "ConvNeXtBase",
]
