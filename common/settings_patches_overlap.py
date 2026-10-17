import torch
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent.parent
DATASET_DIR = BASE_DIR / "dataset_patches_256_overlap"
# TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
# TRAIN_MASK_DIR = DATASET_DIR / "train" / "masks"
# TEST_IMG_DIR = DATASET_DIR / "test" / "images"
# TEST_MASK_DIR = DATASET_DIR / "test" / "masks"
DATASET_IMG_DIR = DATASET_DIR / "images"
DATASET_MASK_DIR = DATASET_DIR / "masks"

TRAIN_MASK_DIR = DATASET_DIR / "masks"

FOLDS_NUM = 3
TEST_SIZE = 20

CONFIDENCE_LEVEL = 0.95


PATCH_SIZE = 256
STRIDE = 128   # <-- как STRIDE_OVERLAP в скрипте нарезки

# Общие параметры
NUM_CLASSES = 5
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 8
NUM_WORKERS = 0   # для Windows
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Цветовая карта (для визуализации)
COLOR_MAP = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 0],
    4: [255, 255, 255]
}

CLASS_NAMES = {
    0: 'background',
    1: 'arteries',
    2: 'veins',
    3: 'crossings',
    4: 'capillaries'
}
