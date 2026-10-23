# common/data_preparation_patches.py
import re
import copy
import random
import typing as tp
from collections import defaultdict
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import settings_patches_overlap as settings
from common.experiment_utils_overlap import strict_rgb_to_labels


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


    def __len__(self) -> int:
        return len(self.image_paths)

    @staticmethod
    def rgb_to_labels(mask_rgb: np.ndarray) -> np.ndarray:
        """
        RGB-маска (H, W, 3) -> индексы классов (H, W) uint8.
        settings.COLOR_MAP: {class_idx: (R, G, B)}.
        """
        return strict_rgb_to_labels(mask_rgb, settings.COLOR_MAP)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]

        try:
            with Image.open(img_path) as opened:
                image = np.array(opened.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать изображение {img_path}: {e}")

        try:
            with Image.open(msk_path) as opened:
                mask_rgb = np.array(opened.convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Не удалось прочитать маску {msk_path}: {e}")

        if image.shape[:2] != mask_rgb.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {img_path}, {msk_path}")
        if image.shape[:2] != (settings.PATCH_SIZE, settings.PATCH_SIZE):
            raise ValueError(f"Expected {settings.PATCH_SIZE}px patch: {img_path}")
        try:
            mask = self.rgb_to_labels(mask_rgb)
        except ValueError as exc:
            raise ValueError(f"{msk_path}: {exc}") from exc

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            ).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        return image.float(), mask.long()


# ===========================================================================
# Группировка патчей по исходному изображению
# ===========================================================================
_PATCH_SUFFIX_RE = re.compile(r"_y\d+_x\d+$")


def _source_key(path: Path) -> str:
    """
    '003_N_y0_x256.png' -> '003_N'
    Некорректный суффикс — ошибка, а не отдельный источник.
    """
    match = _PATCH_SUFFIX_RE.search(path.stem)
    if match is None or match.start() == 0:
        raise ValueError(f"Expected '<source>_y<Y>_x<X>.png': {path.name}")
    return path.stem[:match.start()]


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

    mask_by_stem = {p.stem: p for p in msk_paths}
    return img_paths, [mask_by_stem[p.stem] for p in img_paths]


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
def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    info = torch.utils.data.get_worker_info()
    transform = getattr(info.dataset, "transform", None)
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)


def _compose(transforms, seed):
    transform = A.Compose(copy.deepcopy(transforms))
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(seed)
    return transform


def _generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

class RareClassMixSampler(Sampler):
    """Fixed-length, train-only mixture with source-balanced targeted draws.

    Uniform slots use a shuffled subset without replacement. Targeted slots
    sample a source uniformly, then a positive patch in it. Categories may
    overlap. Patch/source caps include every draw, including uniform slots.
    A missing/exhausted target pool falls back to available training patches.
    No image or mask from the outer validation set is inspected here.
    """

    def __init__(self, image_paths, mask_paths, seed, fractions=None,
                 min_pixels=None, max_patch_repeats=None, max_source_multiplier=None):
        super().__init__()
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths]
        if not self.image_paths or len(self.image_paths) != len(self.mask_paths):
            raise ValueError("Sampler requires nonempty paired training paths.")
        self.seed, self.epoch = int(seed), 0
        self.fractions = dict(settings.RARE_CLASS_FRACTIONS if fractions is None else fractions)
        self.min_pixels = settings.RARE_MIN_PIXELS if min_pixels is None else min_pixels
        self.max_patch_repeats = (settings.RARE_MAX_PATCH_REPEATS
                                  if max_patch_repeats is None else max_patch_repeats)
        self.max_source_multiplier = (settings.RARE_MAX_SOURCE_MULTIPLIER
                                     if max_source_multiplier is None else max_source_multiplier)
        if (not self.fractions or any(c not in range(1, settings.NUM_CLASSES)
                                     for c in self.fractions)
                or any(not np.isfinite(v) or not 0 < v < 1 for v in self.fractions.values())
                or sum(self.fractions.values()) >= 1
                or not isinstance(self.min_pixels, int) or self.min_pixels < 1
                or not isinstance(self.max_patch_repeats, int) or self.max_patch_repeats < 1
                or not np.isfinite(self.max_source_multiplier) or self.max_source_multiplier < 1):
            raise ValueError("Invalid rare-class sampler configuration.")
        self.source_keys = sorted({_source_key(p) for p in self.image_paths})
        source_lookup = {key: i for i, key in enumerate(self.source_keys)}
        self.source_ids = np.array([source_lookup[_source_key(p)] for p in self.image_paths])
        self.source_patch_counts = np.bincount(self.source_ids, minlength=len(self.source_keys))
        self.source_caps = np.ceil(self.source_patch_counts * self.max_source_multiplier).astype(np.int64)
        counts = []
        for path in self.mask_paths:
            with Image.open(path) as opened:
                labels = strict_rgb_to_labels(np.array(opened.convert("RGB")))
            counts.append(np.bincount(labels.ravel(), minlength=settings.NUM_CLASSES))
        self.patch_class_counts = np.asarray(counts, dtype=np.int64)
        self.target_indices = {
            c: np.flatnonzero(self.patch_class_counts[:, c] >= self.min_pixels)
            for c in sorted(self.fractions)
        }
        self.last_epoch_stats = {}
        self.last_source_records = []
        self._cached_epoch, self._cached_indices = None, None
        for c, indices in self.target_indices.items():
            print(f"[rare sampler] class={settings.CLASS_NAMES[c]}: "
                  f"positive patches={len(indices)}, "
                  f"sources={len(np.unique(self.source_ids[indices]))}")
            if not len(indices):
                import warnings
                warnings.warn(f"No eligible training patches for class {c}; use fallback slots.")

    @property
    def protocol_config(self):
        return {
            "name": "rare_class_mixture_v1", "seed": self.seed,
            "fractions": {str(c): float(v) for c, v in sorted(self.fractions.items())},
            "min_pixels": int(self.min_pixels),
            "max_patch_repeats": int(self.max_patch_repeats),
            "max_source_multiplier": float(self.max_source_multiplier),
            "num_samples_per_epoch": len(self),
            "uniform": "random_subset_without_replacement_first",
            "targeted": "uniform_source_then_uniform_positive_patch_with_caps",
            "fallback": "source_balanced_available_training_patches",
            "target_order": sorted(self.fractions),
            "quota_rounding": "floor_each_target_remainder_to_uniform",
        }

    def __len__(self):
        return len(self.image_paths)

    def set_epoch(self, epoch):
        if int(epoch) < 0:
            raise ValueError("Epoch must be nonnegative.")
        self.epoch = int(epoch)

    def audit_records(self):
        rows = []
        for i, path in enumerate(self.image_paths):
            row = {"patch": path.name, "source_key": self.source_keys[self.source_ids[i]]}
            for c in range(settings.NUM_CLASSES):
                row[f"pixels_{settings.CLASS_NAMES[c]}"] = int(self.patch_class_counts[i, c])
            for c in self.target_indices:
                row[f"eligible_{settings.CLASS_NAMES[c]}"] = int(
                    self.patch_class_counts[i, c] >= self.min_pixels)
            rows.append(row)
        return rows

    def __iter__(self):
        # Re-iterating the same epoch is idempotent (e.g. when inspecting a loader).
        if self._cached_epoch == self.epoch:
            return iter(self._cached_indices)
        generator = _generator(self.seed + 1000003 * (self.epoch + 1))
        n = len(self)
        requested = {c: int(n * self.fractions[c]) for c in sorted(self.fractions)}
        uniform_n = n - sum(requested.values())
        chosen = torch.randperm(n, generator=generator)[:uniform_n].tolist()
        patch_draws = np.bincount(chosen, minlength=n).astype(np.int64)
        source_draws = np.bincount(self.source_ids[chosen], minlength=len(self.source_keys)).astype(np.int64)

        def draw_from_pool(indices, amount):
            groups = {}
            for index in indices:
                index = int(index)
                source = int(self.source_ids[index])
                if (patch_draws[index] < self.max_patch_repeats
                        and source_draws[source] < self.source_caps[source]):
                    groups.setdefault(source, []).append(index)
            active_sources = sorted(groups)
            drawn = []
            while len(drawn) < amount and active_sources:
                source_slot = int(torch.randint(len(active_sources), (1,), generator=generator).item())
                source = active_sources[source_slot]
                pool = groups[source]
                slot = int(torch.randint(len(pool), (1,), generator=generator).item())
                index = pool[slot]
                drawn.append(index)
                patch_draws[index] += 1
                source_draws[source] += 1
                if patch_draws[index] >= self.max_patch_repeats:
                    pool[slot] = pool[-1]
                    pool.pop()
                if not pool or source_draws[source] >= self.source_caps[source]:
                    active_sources[source_slot] = active_sources[-1]
                    active_sources.pop()
            return drawn

        delivered = {}
        for c, amount in requested.items():
            targeted = draw_from_pool(self.target_indices[c], amount)
            delivered[c] = len(targeted)
            chosen.extend(targeted)
        fallback_n = n - len(chosen)
        chosen.extend(draw_from_pool(np.arange(n), fallback_n))
        if len(chosen) != n:
            raise RuntimeError("Sampler exhausted total capacity; check repeat/source caps.")
        order = torch.randperm(n, generator=generator).tolist()
        chosen = [chosen[i] for i in order]
        selected = np.asarray(chosen, dtype=np.int64)
        stats = {
            "sampling_draws": n, "sampling_uniform_slots": uniform_n,
            "sampling_fallback_slots": fallback_n,
            "sampling_unique_patches": int(np.count_nonzero(patch_draws)),
            "sampling_unique_fraction": float(np.count_nonzero(patch_draws) / n),
            "sampling_max_patch_repeats": int(patch_draws.max()),
            "sampling_max_source_ratio": float(np.max(source_draws / self.source_patch_counts)),
        }
        for c in range(settings.NUM_CLASSES):
            name = settings.CLASS_NAMES[c]
            stats[f"sampling_positive_fraction_{name}"] = float(
                np.mean(self.patch_class_counts[selected, c] > 0))
            stats[f"sampling_pixels_{name}"] = int(self.patch_class_counts[selected, c].sum())
        for c in requested:
            name = settings.CLASS_NAMES[c]
            stats[f"sampling_requested_{name}"] = requested[c]
            stats[f"sampling_targeted_{name}"] = delivered[c]
        self.last_epoch_stats = stats
        self.last_source_records = []
        for source, key in enumerate(self.source_keys):
            positions = self.source_ids == source
            self.last_source_records.append({
                "epoch": self.epoch + 1, "source_key": key,
                "available_patches": int(self.source_patch_counts[source]),
                "draws": int(source_draws[source]), "draw_cap": int(self.source_caps[source]),
                "unique_patches": int(np.count_nonzero(patch_draws[positions])),
                "max_patch_repeats": int(patch_draws[positions].max()),
            })
        self._cached_epoch, self._cached_indices = self.epoch, chosen
        return iter(chosen)


def _make_train_loader(
    img_paths: List[Path],
    mask_paths: List[Path],
    bs: int,
    nw: int,
    extra_transforms: Optional[List[A.BasicTransform]] = None,
    seed: int = settings.SEED,
) -> DataLoader:
    transforms: List[A.BasicTransform] = [
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
    if extra_transforms:
        transforms.extend(extra_transforms)
    transforms.append(ToTensorV2())
    transform = _compose(transforms, seed)

    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
    sampler = None
    if settings.RARE_SAMPLER_ENABLED:
        sampler = RareClassMixSampler(img_paths, mask_paths, seed=seed)
        # Raw train-only counts, NOT resampled frequencies: keep CE weights unchanged.
        dataset.class_pixel_counts = sampler.patch_class_counts.sum(axis=0).copy()
    return DataLoader(
        dataset, batch_size=bs, shuffle=sampler is None, sampler=sampler, num_workers=nw,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        generator=_generator(seed), worker_init_fn=seed_worker,
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
    transform = _compose(transforms, settings.SEED)

    dataset = SegmentationDataset(img_paths, mask_paths, transform=transform)
    return DataLoader(
        dataset, batch_size=bs, shuffle=False, num_workers=nw,
        pin_memory=torch.cuda.is_available(), drop_last=False,
        generator=_generator(settings.SEED), worker_init_fn=seed_worker,
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
    settings.validate_settings()
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

    bs = settings.BATCH_SIZE if batch_size is None else batch_size
    nw = settings.NUM_WORKERS if num_workers is None else num_workers
    if bs < 1 or nw < 0:
        raise ValueError("Invalid batch size or worker count.")

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
        for k in sorted(train_keys):
            i_list, m_list = groups[k]
            train_imgs.extend(i_list)
            train_masks.extend(m_list)

        val_imgs: List[Path] = []
        val_masks: List[Path] = []
        for k in sorted(val_keys):
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
            train_imgs, train_masks, bs, nw, train_extra_transforms,
            seed=settings.SEED + fold + 1,
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
