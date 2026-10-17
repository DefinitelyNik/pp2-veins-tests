# common/data_preparation_patches.py
import re
import typing as tp
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import settings_patches as settings


# ===========================================================================
# Dataset
# ===========================================================================
class SegmentationDataset(Dataset):
    def __init__(self, image_paths: List[Path], mask_paths: List[Path],
                 transform: Optional[A.Compose] = None):
        assert len(image_paths) == len(mask_paths), (
            f"Число изображений ({len(image_paths)}) и масок ({len(mask_paths)}) "
            f"не совпадает"
        )
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def rgb_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = mask_rgb.shape
        mask_labels = np.zeros((h, w), dtype=np.uint8)
        for class_idx, color in settings.COLOR_MAP.items():
            color_mask = np.all(mask_rgb == np.array(color, dtype=mask_rgb.dtype),
                                axis=-1)
            mask_labels[color_mask] = class_idx
        return mask_labels

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_paths[idx]).convert('RGB'))
        mask = self.rgb_to_labels(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.long()
        return image, mask


# ===========================================================================
# Группировка патчей по исходному изображению
# ===========================================================================
_PATCH_SUFFIX_RE = re.compile(r"_y\d+_x\d+$")


def _source_key(path: Path) -> str:
    """
    '003_N_y0_x256' -> '003_N'
    Если суффикса нет — вернёт stem как есть.
    """
    return _PATCH_SUFFIX_RE.sub("", path.stem)


def _group_patches_by_source(img_paths: List[Path], mask_paths: List[Path]):
    assert len(img_paths) == len(mask_paths)
    groups: tp.Dict[str, tp.Tuple[List[Path], List[Path]]] = defaultdict(
        lambda: ([], [])
    )
    for ip, mp in zip(img_paths, mask_paths):
        key = _source_key(ip)
        groups[key][0].append(ip)
        groups[key][1].append(mp)
    return groups


# ===========================================================================
# Валидация
# ===========================================================================
def _validate_dataset():
    img_dir = settings.DATASET_IMG_DIR
    msk_dir = settings.DATASET_MASK_DIR

    assert img_dir.is_dir(), f"Нет папки с изображениями: {img_dir}"
    assert msk_dir.is_dir(), f"Нет папки с масками: {msk_dir}"

    img_paths = sorted(img_dir.glob("*.png"))
    msk_paths = sorted(msk_dir.glob("*.png"))

    assert len(img_paths) > 0, f"В {img_dir} не найдено ни одного .png"
    assert len(msk_paths) > 0, f"В {msk_dir} не найдено ни одного .png"

    img_stems = {p.stem for p in img_paths}
    msk_stems = {p.stem for p in msk_paths}
    only_img = img_stems - msk_stems
    only_msk = msk_stems - img_stems
    assert not only_img and not only_msk, (
        f"Рассинхрон images/masks: только в images={len(only_img)}, "
        f"только в masks={len(only_msk)}. Пример: "
        f"{sorted(only_img)[:3]} vs {sorted(only_msk)[:3]}"
    )
    return img_paths, msk_paths


def _validate_settings_for_cv(source_keys: List[str]) -> None:
    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE
    total = len(source_keys)

    assert num_folds >= 2, f"FOLDS_NUM={num_folds} — нужно минимум 2"
    assert test_size >= 1, f"TEST_SIZE={test_size} — должно быть >= 1"
    assert total >= num_folds * test_size, (
        f"Исходных изображений {total}, требуется "
        f"{num_folds * test_size} ({num_folds} фолдов × {test_size}). "
        f"Уменьшите FOLDS_NUM или TEST_SIZE."
    )
    assert total - test_size > 0, (
        f"train пустой: всего источников {total}, TEST_SIZE={test_size}"
    )


# ===========================================================================
# Лоадеры
# ===========================================================================
def _make_train_loader(img_paths, mask_paths, bs, nw, extra):
    transforms: List[A.BasicTransform] = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
    ]
    if extra:
        transforms.extend(extra)
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)
    ds = SegmentationDataset(img_paths, mask_paths, transform=transform)
    return DataLoader(ds, batch_size=bs, shuffle=True,
                      num_workers=nw, pin_memory=True, drop_last=False)


def _make_val_loader(img_paths, mask_paths, bs, nw, extra):
    transforms: List[A.BasicTransform] = []
    if extra:
        transforms.extend(extra)
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)
    ds = SegmentationDataset(img_paths, mask_paths, transform=transform)
    return DataLoader(ds, batch_size=bs, shuffle=False,
                      num_workers=nw, pin_memory=True, drop_last=False)


# ===========================================================================
# Кросс-валидация по источникам (без утечки)
# ===========================================================================
def create_cross_val_loaders(
    train_extra_transforms: Optional[List[A.BasicTransform]] = None,
    val_extra_transforms: Optional[List[A.BasicTransform]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[tp.Tuple[DataLoader, DataLoader]]:
    img_paths, msk_paths = _validate_dataset()
    groups = _group_patches_by_source(img_paths, msk_paths)
    source_keys = sorted(groups.keys())
    _validate_settings_for_cv(source_keys)

    rng = np.random.RandomState(settings.SEED)
    shuffled_keys = source_keys.copy()
    rng.shuffle(shuffled_keys)

    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE
    bs = batch_size or settings.BATCH_SIZE
    nw = num_workers or settings.NUM_WORKERS

    if verbose:
        print(f"[CV] источников={len(source_keys)}, патчей={len(img_paths)}, "
              f"фолдов={num_folds}, val-источников на фолд={test_size}, "
              f"batch_size={bs}, workers={nw}")

    result = []
    for fold in range(num_folds):
        start = fold * test_size
        end = start + test_size

        val_keys = set(shuffled_keys[start:end])
        train_keys = set(shuffled_keys) - val_keys

        assert not (train_keys & val_keys), f"Утечка на фолде {fold}"

        train_imgs, train_masks = [], []
        for k in train_keys:
            i_list, m_list = groups[k]
            train_imgs.extend(i_list)
            train_masks.extend(m_list)

        val_imgs, val_masks = [], []
        for k in val_keys:
            i_list, m_list = groups[k]
            val_imgs.extend(i_list)
            val_masks.extend(m_list)

        assert len(train_imgs) == len(train_masks)
        assert len(val_imgs) == len(val_masks)
        assert len(train_imgs) > 0 and len(val_imgs) > 0

        train_loader = _make_train_loader(train_imgs, train_masks, bs, nw,
                                          train_extra_transforms)
        val_loader = _make_val_loader(val_imgs, val_masks, bs, nw,
                                      val_extra_transforms)
        result.append((train_loader, val_loader))

        if verbose:
            print(f"[CV] fold {fold}: train_src={len(train_keys)}, "
                  f"val_src={len(val_keys)}, "
                  f"train_patches={len(train_imgs)}, "
                  f"val_patches={len(val_imgs)}")

    _check_no_leak(result)
    return result


def _check_no_leak(loaders) -> None:
    for fold, (tr, vl) in enumerate(loaders):
        tr_keys = {_source_key(Path(p)) for p in tr.dataset.image_paths}
        vl_keys = {_source_key(Path(p)) for p in vl.dataset.image_paths}
        assert not (tr_keys & vl_keys), (
            f"Утечка на фолде {fold}: {sorted(tr_keys & vl_keys)[:5]}"
        )


def get_val_source_keys_per_fold() -> List[List[str]]:
    """
    Список val-источников по фолдам — согласован с create_cross_val_loaders
    (тот же SEED и порядок).
    """
    img_paths, msk_paths = _validate_dataset()
    groups = _group_patches_by_source(img_paths, msk_paths)
    source_keys = sorted(groups.keys())
    _validate_settings_for_cv(source_keys)

    rng = np.random.RandomState(settings.SEED)
    shuffled_keys = source_keys.copy()
    rng.shuffle(shuffled_keys)

    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE

    return [
        shuffled_keys[f * test_size:(f + 1) * test_size]
        for f in range(num_folds)
    ]


# ===========================================================================
# Самопроверка
# ===========================================================================
if __name__ == "__main__":
    loaders = create_cross_val_loaders()
    for fold, (tr, vl) in enumerate(loaders):
        images, masks = next(iter(tr))
        print(f"fold {fold}: batch images {tuple(images.shape)} "
              f"{images.dtype}, masks {tuple(masks.shape)} {masks.dtype}")
    print("OK: подготовка данных корректна, утечки нет")
