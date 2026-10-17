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

from common import settings_patches_overlap as settings


# ===========================================================================
# Dataset
# ===========================================================================
class SegmentationDataset(Dataset):
    """
    Изображение: RGB (H, W, 3).
    Маска на диске: цветная (H, W, 3) в RGB.
    На выходе: image (3, H, W) float, mask (H, W) long (индексы классов).
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[A.Compose] = None,
    ):
        assert len(image_paths) == len(mask_paths), (
            f"Число изображений ({len(image_paths)}) и масок ({len(mask_paths)}) "
            f"не совпадает"
        )
        self.image_paths = list(image_paths)
        self.mask_paths = list(mask_paths)
        self.transform = transform
        self._checked_once = False

    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def rgb_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
        """
        RGB-маска (H, W, 3) -> индексы классов (H, W) uint8.
        settings.COLOR_MAP: {class_idx: (R, G, B)}.
        """
        assert mask_rgb.ndim == 3 and mask_rgb.shape[2] == 3, \
            f"Ожидали RGB-маску (H,W,3), получили {mask_rgb.shape}"
        h, w, _ = mask_rgb.shape
        mask_labels = np.zeros((h, w), dtype=np.uint8)
        for class_idx, color in settings.COLOR_MAP.items():
            color_arr = np.array(color, dtype=mask_rgb.dtype)
            color_mask = np.all(mask_rgb == color_arr, axis=-1)
            mask_labels[color_mask] = class_idx
        return mask_labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать изображение {img_path}: {e}")

        try:
            mask_rgb = np.array(Image.open(msk_path).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать маску {msk_path}: {e}")

        mask = self.rgb_to_labels(mask_rgb)

        # Разовая проверка: нет ли в маске классов, которых нет в COLOR_MAP
        if not self._checked_once:
            uniq = set(np.unique(mask).tolist())
            known = set(settings.COLOR_MAP.keys())
            unknown = uniq - known
            assert not unknown, (
                f"В маске {msk_path} найдены классы {unknown}, "
                f"которых нет в COLOR_MAP {sorted(known)}"
            )
            self._checked_once = True

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.long()
        return image, mask


# ===========================================================================
# Группировка патчей по исходному изображению
# ===========================================================================
_PATCH_SUFFIX_RE = re.compile(r"_y\d+_x\d+$")


def _source_key(path: Path) -> str:
    """
    '003_N_y0_x256.png' -> '003_N'
    Если суффикса нет — возвращает stem как есть.
    """
    return _PATCH_SUFFIX_RE.sub("", path.stem)


def _group_patches_by_source(
    img_paths: List[Path],
    mask_paths: List[Path],
) -> tp.Dict[str, tp.Tuple[List[Path], List[Path]]]:
    assert len(img_paths) == len(mask_paths), \
        "Число изображений и масок не совпадает"

    groups: tp.Dict[str, tp.Tuple[List[Path], List[Path]]] = defaultdict(
        lambda: ([], [])
    )
    for ip, mp in zip(img_paths, mask_paths):
        key = _source_key(ip)
        groups[key][0].append(ip)
        groups[key][1].append(mp)
    return groups


# ===========================================================================
# Проверки подготовки данных
# ===========================================================================
def _validate_dataset() -> tp.Tuple[List[Path], List[Path]]:
    """
    Собирает пути, проверяет парность images/masks, непустоту.
    """
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
        f"Рассинхрон images/masks: "
        f"только в images={len(only_img)}, только в masks={len(only_msk)}. "
        f"Пример: только images -> {sorted(only_img)[:3]}, "
        f"только masks -> {sorted(only_msk)[:3]}"
    )

    assert len(img_paths) == len(msk_paths), (
        f"len(images)={len(img_paths)} != len(masks)={len(msk_paths)}"
    )

    return img_paths, msk_paths


def _validate_settings_for_cv(source_keys: List[str]) -> None:
    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE
    total_sources = len(source_keys)

    assert num_folds >= 2, f"FOLDS_NUM={num_folds} — нужно минимум 2"
    assert test_size >= 1, f"TEST_SIZE={test_size} — должно быть >= 1"

    assert total_sources >= num_folds * test_size, (
        f"Исходных изображений {total_sources}, требуется "
        f"{num_folds * test_size} ({num_folds} фолдов × {test_size}). "
        f"Уменьшите FOLDS_NUM или TEST_SIZE."
    )

    assert total_sources - test_size > 0, (
        f"train пустой: всего источников {total_sources}, "
        f"TEST_SIZE={test_size}"
    )


# ===========================================================================
# Лоадеры
# ===========================================================================
def _make_train_loader(
    img_paths: List[Path],
    mask_paths: List[Path],
    bs: int,
    nw: int,
    extra_transforms: Optional[List[A.BasicTransform]] = None,
) -> DataLoader:
    transforms: List[A.BasicTransform] = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if extra_transforms:
        transforms.extend(extra_transforms)
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)

    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )


def _make_val_loader(
    img_paths: List[Path],
    mask_paths: List[Path],
    bs: int,
    nw: int,
    extra_transforms: Optional[List[A.BasicTransform]] = None,
) -> DataLoader:
    transforms: List[A.BasicTransform] = []
    if extra_transforms:
        transforms.extend(extra_transforms)
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)

    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        drop_last=False,
    )


# ===========================================================================
# Кросс-валидация без утечки
# ===========================================================================
def create_cross_val_loaders(
    train_extra_transforms: Optional[List[A.BasicTransform]] = None,
    val_extra_transforms: Optional[List[A.BasicTransform]] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[tp.Tuple[DataLoader, DataLoader]]:
    """
    Кросс-валидация с группировкой по исходным изображениям.
    Патчи одного исходного изображения не могут попасть одновременно
    в train и val.

    settings.TEST_SIZE — число ИСХОДНЫХ изображений в val-фолде.
    """
    img_paths, msk_paths = _validate_dataset()

    groups = _group_patches_by_source(img_paths, msk_paths)
    source_keys = sorted(groups.keys())
    total_sources = len(source_keys)

    _validate_settings_for_cv(source_keys)

    rng = np.random.RandomState(settings.SEED)
    shuffled_keys = source_keys.copy()
    rng.shuffle(shuffled_keys)

    num_folds = settings.FOLDS_NUM
    test_size = settings.TEST_SIZE

    bs = batch_size or settings.BATCH_SIZE
    nw = num_workers or settings.NUM_WORKERS

    if verbose:
        print(f"[CV] источников={total_sources}, патчей={len(img_paths)}, "
              f"фолдов={num_folds}, val-источников на фолд={test_size}, "
              f"batch_size={bs}, workers={nw}")

    result: List[tp.Tuple[DataLoader, DataLoader]] = []
    for fold in range(num_folds):
        start = fold * test_size
        end = start + test_size

        val_keys = set(shuffled_keys[start:end])
        train_keys = set(shuffled_keys) - val_keys

        overlap = train_keys & val_keys
        assert not overlap, (
            f"Утечка на фолде {fold}: пересечение источников: "
            f"{sorted(overlap)[:5]}"
        )
        assert len(train_keys) > 0, f"Пустой train на фолде {fold}"
        assert len(val_keys) > 0, f"Пустой val на фолде {fold}"

        train_imgs: List[Path] = []
        train_masks: List[Path] = []
        for k in train_keys:
            i_list, m_list = groups[k]
            train_imgs.extend(i_list)
            train_masks.extend(m_list)

        val_imgs: List[Path] = []
        val_masks: List[Path] = []
        for k in val_keys:
            i_list, m_list = groups[k]
            val_imgs.extend(i_list)
            val_masks.extend(m_list)

        assert len(train_imgs) == len(train_masks), \
            f"fold {fold}: train imgs/masks рассинхрон"
        assert len(val_imgs) == len(val_masks), \
            f"fold {fold}: val imgs/masks рассинхрон"
        assert len(train_imgs) > 0, f"fold {fold}: пустой train_imgs"
        assert len(val_imgs) > 0, f"fold {fold}: пустой val_imgs"

        train_loader = _make_train_loader(
            train_imgs, train_masks, bs, nw, train_extra_transforms
        )
        val_loader = _make_val_loader(
            val_imgs, val_masks, bs, nw, val_extra_transforms
        )
        result.append((train_loader, val_loader))

        if verbose:
            print(f"[CV] fold {fold}: "
                  f"train_src={len(train_keys)}, val_src={len(val_keys)}, "
                  f"train_patches={len(train_imgs)}, "
                  f"val_patches={len(val_imgs)}")

    _check_no_leak(result)
    return result


def _check_no_leak(loaders: List[tp.Tuple[DataLoader, DataLoader]]) -> None:
    for fold, (tr, vl) in enumerate(loaders):
        tr_keys = {_source_key(Path(p)) for p in tr.dataset.image_paths}
        vl_keys = {_source_key(Path(p)) for p in vl.dataset.image_paths}
        overlap = tr_keys & vl_keys
        assert not overlap, (
            f"Утечка на фолде {fold}: {sorted(overlap)[:5]}"
        )


# ===========================================================================
# Дополнительно: получить список val-источников по фолду (пригодится для метрик)
# ===========================================================================
def get_val_source_keys_per_fold() -> List[List[str]]:
    """
    Возвращает для каждого фолда список source_key, попавших в val.
    Должно совпадать с порядком фолдов в create_cross_val_loaders
    (тот же SEED, тот же shuffle).
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

    result: List[List[str]] = []
    for fold in range(num_folds):
        start = fold * test_size
        end = start + test_size
        result.append(shuffled_keys[start:end])
    return result


# ===========================================================================
# Самопроверка при запуске модуля
# ===========================================================================
if __name__ == "__main__":
    loaders = create_cross_val_loaders()
    for fold, (tr, vl) in enumerate(loaders):
        images, masks = next(iter(tr))
        uniq = torch.unique(masks).tolist()
        assert all(0 <= u < len(settings.COLOR_MAP) for u in uniq), (
            f"fold {fold}: метки маски выходят за пределы COLOR_MAP: {uniq}"
        )
        print(
            f"fold {fold}: batch images {tuple(images.shape)} {images.dtype}, "
            f"masks {tuple(masks.shape)} {masks.dtype}, "
            f"unique classes {uniq}"
        )
    print("OK: подготовка данных корректна, утечки нет")
