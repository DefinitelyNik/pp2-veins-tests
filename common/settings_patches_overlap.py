"""Shared configuration. Place this file in common/."""
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset_patches_256_overlap"
DATASET_IMG_DIR = DATASET_DIR / "images"
DATASET_MASK_DIR = DATASET_DIR / "masks"
NORMALIZED_IMG_DIR = BASE_DIR / "normalized_dataset" / "images"
NORMALIZED_MASK_DIR = BASE_DIR / "normalized_dataset" / "masks"

# Separate output directory: never overwrite the original baseline weights.
OUTPUT_DIR = BASE_DIR / "UNETPP_WCD" / "runs" / "vessel_aware_50ep"

# Preserve the ORIGINAL three splits, not a full 5-fold CV.
FOLDS_NUM = 3
N_FOLDS = FOLDS_NUM  # compatibility alias, not an independent setting
TEST_SIZE = 20
SEED = 42
CONFIDENCE_LEVEL = 0.95
EPOCHS_COUNT = 50
PATCH_SIZE = 256
STRIDE = 128
NUM_CLASSES = 5
BATCH_SIZE = 8
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"  # set None for a fully offline run
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
UNION_COEF = 0.25
TOPOLOGY_COEF = 0.10
SKELETON_ITERATIONS = 10
USE_PHOTOMETRIC_AUG = True
USE_TTA = False
BOOTSTRAP_SAMPLES = 2000

# Colors are preserved from the supplied project. Verify against your archive.
COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0],
    4: [255, 255, 255],
}
# Original Fundus-AVSeg uses 'uncertain vessels', not anatomical capillaries.
# If your dataset is a reannotated variant, adjust the name to its documentation.
CLASS_NAMES = {
    0: "background",
    1: "arteries",
    2: "veins",
    3: "crossings",
    4: "uncertain_vessels",
}

# Optional CSV columns: source_key, patient_id.
# If several images belong to one patient, supply this file.
# The original source splits are preserved; an overlapping patient raises an error.
PATIENT_GROUPS_CSV = None


def validate_settings():
    if NUM_CLASSES != 5 or set(COLOR_MAP) != set(range(NUM_CLASSES)):
        raise ValueError("Expected five contiguous class indices 0..4.")
    colors = [tuple(c) for c in COLOR_MAP.values()]
    if len(set(colors)) != NUM_CLASSES:
        raise ValueError("Duplicate colors in COLOR_MAP.")
    if any(len(c) != 3 or any(not 0 <= v <= 255 for v in c) for c in colors):
        raise ValueError("COLOR_MAP must contain RGB triplets in 0..255.")
    if set(CLASS_NAMES) != set(COLOR_MAP):
        raise ValueError("CLASS_NAMES and COLOR_MAP keys must match.")
    if EPOCHS_COUNT != 50:
        raise ValueError("This experiment requires exactly 50 epochs.")
    if PATCH_SIZE < 32 or PATCH_SIZE % 32:
        raise ValueError("PATCH_SIZE must be a positive multiple of 32.")
    if not 0 < STRIDE <= PATCH_SIZE:
        raise ValueError("Require 0 < STRIDE <= PATCH_SIZE.")
    if BATCH_SIZE < 1 or NUM_WORKERS < 0:
        raise ValueError("Invalid batch size or worker count.")
    if FOLDS_NUM < 2 or TEST_SIZE < 1:
        raise ValueError("Invalid CV settings.")
    if UNION_COEF < 0 or TOPOLOGY_COEF < 0 or SKELETON_ITERATIONS < 1:
        raise ValueError("Invalid loss settings.")
