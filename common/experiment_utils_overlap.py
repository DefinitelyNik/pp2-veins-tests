"""Place in common/experiment_utils_overlap.py. No model imports."""
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from common import settings_patches_overlap as settings


def strict_rgb_to_labels(mask_rgb, color_map=None):
    color_map = settings.COLOR_MAP if color_map is None else color_map
    if mask_rgb.ndim != 3 or mask_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB mask, got {mask_rgb.shape}")
    labels = np.full(mask_rgb.shape[:2], -1, dtype=np.int16)
    for index, color in color_map.items():
        labels[np.all(mask_rgb == np.asarray(color), axis=-1)] = index
    bad = labels < 0
    if bad.any():
        colors = np.unique(mask_rgb[bad], axis=0)[:10].tolist()
        raise ValueError(
            f"Unknown mask colors: {colors}; pixels={int(bad.sum())}. "
            "Verify COLOR_MAP. Resize masks with nearest-neighbor only."
        )
    return labels.astype(np.uint8)


def read_rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def seed_everything(seed):
    # Set before first CUDA computation; setting PYTHONHASHSEED here would not
    # change the running interpreter. All important set traversals are sorted.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def patient_mapping(source_keys):
    if settings.PATIENT_GROUPS_CSV is None:
        return {key: key for key in source_keys}
    path = Path(settings.PATIENT_GROUPS_CSV)
    if not path.is_absolute():
        path = settings.BASE_DIR / path
    mapping = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"source_key", "patient_id"}.issubset(reader.fieldnames or []):
            raise ValueError("Patient CSV requires source_key,patient_id columns.")
        for row in reader:
            key, patient = row["source_key"].strip(), row["patient_id"].strip()
            if not key or not patient or key in mapping:
                raise ValueError(f"Empty or duplicate patient CSV entry: {row}")
            mapping[key] = patient
    missing = set(source_keys) - set(mapping)
    if missing:
        raise ValueError(f"Patient CSV missing sources: {sorted(missing)[:10]}")
    return {key: mapping[key] for key in source_keys}


def validate_patient_splits(loaders, mapping):
    from common.data_preparation_patches_overlap import _source_key
    for number, (train, val) in enumerate(loaders, 1):
        train_groups = {mapping[_source_key(p)] for p in train.dataset.image_paths}
        val_groups = {mapping[_source_key(p)] for p in val.dataset.image_paths}
        overlap = train_groups & val_groups
        if overlap:
            raise ValueError(
                f"Fold {number}: patient leakage {sorted(overlap)[:10]}. "
                "Original splits cannot be preserved safely; use patient-level "
                "splits and rerun ALL comparison models."
            )


def preflight_dataset(verbose=True):
    """Full audit before training, including exact source/patch consistency.

    Expected patches are crops of normalized_dataset with NO separate color
    processing. A mismatch is a data-pipeline error, never silently corrected.
    Hashes include source and patch file content, not timestamps.
    """
    from common.data_preparation_patches_overlap import (
        _validate_dataset, _group_patches_by_source,
    )
    from common.metrics_patches_overlap import parse_patch_name

    settings.validate_settings()
    images, masks = _validate_dataset()
    groups = _group_patches_by_source(images, masks)
    digest = hashlib.sha256()
    counts = np.zeros(settings.NUM_CLASSES, dtype=np.int64)

    def add_hash(label, path):
        digest.update((label + ":" + file_sha256(path) + "\n").encode())

    for source in sorted(groups):
        image_path = settings.NORMALIZED_IMG_DIR / f"{source}.png"
        mask_path = settings.NORMALIZED_MASK_DIR / f"{source}.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing normalized image/mask pair: {source}")
        image, rgb_mask = read_rgb(image_path), read_rgb(mask_path)
        if image.shape != rgb_mask.shape:
            raise ValueError(f"Source image/mask dimensions differ: {source}")
        strict_rgb_to_labels(rgb_mask)
        add_hash("full_image/" + source, image_path)
        add_hash("full_mask/" + source, mask_path)
        source_images, source_masks = groups[source]
        for ip, mp in zip(source_images, source_masks):
            patch_image, patch_mask = read_rgb(ip), read_rgb(mp)
            expected = (settings.PATCH_SIZE, settings.PATCH_SIZE, 3)
            if patch_image.shape != expected or patch_mask.shape != expected:
                raise ValueError(f"Expected shape {expected}: {ip}, {mp}")
            labels = strict_rgb_to_labels(patch_mask)
            counts += np.bincount(labels.ravel(), minlength=settings.NUM_CLASSES)
            _, y, x = parse_patch_name(ip.stem)
            p = settings.PATCH_SIZE
            if y + p > image.shape[0] or x + p > image.shape[1]:
                raise ValueError(f"Patch extends outside source image: {ip}")
            if not np.array_equal(patch_mask, rgb_mask[y:y+p, x:x+p]):
                raise ValueError(
                    f"Patch mask does not equal its normalized source crop: {mp}. "
                    "Rebuild patches from the normalized sources without mask interpolation."
                )
            if not np.array_equal(patch_image, image[y:y+p, x:x+p]):
                raise ValueError(
                    f"Patch image does not equal its normalized source crop: {ip}. "
                    "Do not apply independent offline normalization to patches."
                )
            add_hash("patch_image/" + ip.name, ip)
            add_hash("patch_mask/" + mp.name, mp)
        if verbose:
            print(f"[audit] {source}: {len(source_images)} patches OK")
    mapping = patient_mapping(sorted(groups))
    if settings.PATIENT_GROUPS_CSV is None:
        print("[audit] No patient CSV: independence is checked at source-image level only.")
    return {
        "dataset_sha256": digest.hexdigest(),
        "sources": sorted(groups),
        "patches": len(images),
        "all_patch_class_counts": counts.tolist(),
        "patient_mapping": mapping,
    }


def fold_manifest(train_loader, val_loader, fold, audit):
    from common.data_preparation_patches_overlap import _source_key
    train_sources = sorted({_source_key(p) for p in train_loader.dataset.image_paths})
    val_sources = sorted({_source_key(p) for p in val_loader.dataset.image_paths})
    payload = {
        "schema": 1,
        "fold": fold,
        "seed": settings.SEED + fold,
        "epochs": settings.EPOCHS_COUNT,
        "selection": "fixed_epoch_50_no_outer_validation_selection",
        "dataset_sha256": audit["dataset_sha256"],
        "train_sources": train_sources,
        "val_sources": val_sources,
        "patient_mapping": audit["patient_mapping"],
        "batch_size": settings.BATCH_SIZE,
        "patch_size": settings.PATCH_SIZE,
        "stride": settings.STRIDE,
        "color_map": {str(k): v for k, v in settings.COLOR_MAP.items()},
        "class_names": {str(k): v for k, v in settings.CLASS_NAMES.items()},
        "encoder": settings.ENCODER_NAME,
        "encoder_weights": settings.ENCODER_WEIGHTS,
        "learning_rate": settings.LEARNING_RATE,
        "weight_decay": settings.WEIGHT_DECAY,
        "union_coef": settings.UNION_COEF,
        "topology_coef": settings.TOPOLOGY_COEF,
        "skeleton_iterations": settings.SKELETON_ITERATIONS,
        "photometric_augmentation": settings.USE_PHOTOMETRIC_AUG,
        "mean": list(settings.MEAN),
        "std": list(settings.STD),
        "num_workers": settings.NUM_WORKERS,
        "train_batches_per_epoch": len(train_loader),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["protocol_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload
