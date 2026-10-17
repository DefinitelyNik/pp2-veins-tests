# common/metrics_patches_overlap.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

from common import settings_patches_overlap as settings


# ---------------------------------------------------------------------------
# Парсинг имени патча
# ---------------------------------------------------------------------------
_PATCH_NAME_RE = re.compile(r"^(?P<source>.+)_y(?P<y>\d+)_x(?P<x>\d+)$")


def parse_patch_name(stem: str) -> Tuple[str, int, int]:
    """
    '001_G_y0_x256' -> ('001_G', 0, 256)
    """
    m = _PATCH_NAME_RE.match(stem)
    if not m:
        raise ValueError(
            f"Не удалось распарсить имя патча: {stem}. "
            f"Ожидали '<source>_y<Y>_x<X>'"
        )
    return m.group("source"), int(m.group("y")), int(m.group("x"))


def color_mask_to_index(mask_rgb: np.ndarray, color_map: Dict[int, list]) -> np.ndarray:
    """RGB-маска (H,W,3) -> индексы классов (H,W) uint8."""
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in color_map.items():
        arr = np.array(color, dtype=mask_rgb.dtype)
        m = np.all(mask_rgb == arr, axis=-1)
        out[m] = cls
    return out


# ---------------------------------------------------------------------------
# Нарезка изображения на патчи прямо в памяти
# ---------------------------------------------------------------------------
def iter_patch_coords(H: int, W: int, patch: int, stride: int):
    """
    Генерирует координаты (y, x) верхнего левого угла патчей,
    покрывающих изображение HxW. Логика та же, что в скрипте нарезки:
    последний патч добивается до края, если сетка не попадает точно.
    """
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    for y in ys:
        for x in xs:
            yield y, x


# ---------------------------------------------------------------------------
# Восстановление из патчей
# ---------------------------------------------------------------------------
def reconstruct_from_patches(
    patch_preds: List[np.ndarray],
    patch_names: List[str],
    full_h: int,
    full_w: int,
    num_classes: int,
    priority_order: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Восстанавливает полное изображение из overlap-патчей.

    patch_preds : список (ph, pw) uint8 с предсказанными классами
    patch_names : список имён (stem) в формате <source>_y<Y>_x<X>
    priority_order : порядок приоритета при равенстве голосов.
                     Первый в списке — самый приоритетный.
                     По умолчанию: [3, 1, 2, 4, 0]

    Возвращает (full_pred, coverage):
        full_pred : (full_h, full_w) uint8
        coverage  : (full_h, full_w) int32 — сколько патчей покрыли пиксель
    """
    if priority_order is None:
        priority_order = [3, 1, 2, 4, 0]
    assert set(priority_order) == set(range(num_classes)), \
        f"priority_order должен содержать все {num_classes} классов"

    votes = np.zeros((num_classes, full_h, full_w), dtype=np.int32)
    coverage = np.zeros((full_h, full_w), dtype=np.int32)

    for pred, name in zip(patch_preds, patch_names):
        _, y, x = parse_patch_name(name)
        ph, pw = pred.shape

        y_end = min(y + ph, full_h)
        x_end = min(x + pw, full_w)
        ph_ = y_end - y
        pw_ = x_end - x
        if ph_ <= 0 or pw_ <= 0:
            continue

        pred_crop = pred[:ph_, :pw_]
        for c in range(num_classes):
            votes[c, y:y_end, x:x_end] += (pred_crop == c).astype(np.int32)
        coverage[y:y_end, x:x_end] += 1

    # Перестановка каналов в порядке приоритета: argmax вернёт
    # индекс в переставленном массиве — соответствует priority_order[idx].
    votes_perm = votes[priority_order, :, :]
    idx = np.argmax(votes_perm, axis=0)
    full_pred = np.array(priority_order, dtype=np.uint8)[idx]

    return full_pred, coverage


# ---------------------------------------------------------------------------
# Confusion matrix и метрики
# ---------------------------------------------------------------------------
def _confusion_matrix_np(preds: np.ndarray, targets: np.ndarray,
                         num_classes: int) -> np.ndarray:
    """Быстрая confusion matrix без sklearn."""
    mask = (targets >= 0) & (targets < num_classes) & \
           (preds >= 0) & (preds < num_classes)
    idx = targets[mask].astype(np.int64) * num_classes + preds[mask].astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_macro_dice_iou(cm: np.ndarray) -> Tuple[float, float]:
    num_classes = cm.shape[0]
    dice, iou = [], []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou.append(tp / (tp + fp + fn + 1e-8))
    return float(np.mean(dice)), float(np.mean(iou))


def compute_per_class_dice_iou(cm: np.ndarray,
                               class_names: Dict[int, str]) -> Dict[str, float]:
    num_classes = cm.shape[0]
    result = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        name = class_names.get(cls, f"class_{cls}")
        result[f"{name}_dice"] = dice
        result[f"{name}_iou"] = iou
    return result


def compute_per_class_accuracy_precision_recall(
    cm: np.ndarray, class_names: Dict[int, str]
) -> Dict[str, float]:
    num_classes = cm.shape[0]
    total = cm.sum()
    result = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        tn = total - tp - fp - fn
        acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        name = class_names.get(cls, f"class_{cls}")
        result[f"{name}_accuracy"] = acc
        result[f"{name}_precision"] = prec
        result[f"{name}_recall"] = rec
    return result


def get_all_metrics_from_cm(cm: np.ndarray,
                            class_names: Dict[int, str]) -> Dict[str, float]:
    per_class = compute_per_class_dice_iou(cm, class_names)
    per_class.update(compute_per_class_accuracy_precision_recall(cm, class_names))
    return per_class


# ---------------------------------------------------------------------------
# Оценка фолда по восстановленным изображениям
# ---------------------------------------------------------------------------
def evaluate_fold_overlap(
    model,
    val_source_keys: List[str],
    normalized_img_dir: Path,
    normalized_msk_dir: Path,
    num_classes: int,
    color_map: Dict[int, list],
    class_names: Dict[int, str],
    val_extra_transforms,                # <-- новый аргумент
    patch_size: int = 256,
    stride: int = 128,
    priority_order: Optional[List[int]] = None,
    batch_size: int = 8,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    import torch

    if priority_order is None:
        priority_order = [3, 1, 2, 4, 0]

    # Тот же transform, что и в _make_val_loader
    val_transform = A.Compose([*val_extra_transforms, ToTensorV2()])

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    visualizations: Dict[str, np.ndarray] = {}

    for src in val_source_keys:
        img_path = normalized_img_dir / f"{src}.png"
        msk_path = normalized_msk_dir / f"{src}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Нет исходного изображения {img_path}")
        if not msk_path.exists():
            raise FileNotFoundError(f"Нет исходной маски {msk_path}")

        img_rgb = np.array(Image.open(img_path).convert("RGB"))  # (H,W,3) uint8
        gt_rgb = np.array(Image.open(msk_path).convert("RGB"))
        H, W = img_rgb.shape[:2]
        if gt_rgb.shape[:2] != (H, W):
            raise RuntimeError(
                f"{src}: размер изображения {img_rgb.shape[:2]} != "
                f"размер маски {gt_rgb.shape[:2]}"
            )
        gt_idx = color_mask_to_index(gt_rgb, color_map)

        coords = list(iter_patch_coords(H, W, patch_size, stride))

        patch_names: List[str] = []
        patch_preds: List[np.ndarray] = []

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]

            batch_imgs = []
            for (y, x) in batch_coords:
                im = img_rgb[y:y + patch_size, x:x + patch_size]
                batch_imgs.append(im)
            batch_arr = np.stack(batch_imgs, axis=0)  # (B, ps, ps, 3) uint8

            # --- прогон через тот же transform, что в val_loader ---
            tensors = []
            for b in range(batch_arr.shape[0]):
                out = val_transform(image=batch_arr[b])
                tensors.append(out["image"])            # (3, ps, ps) float32
            tensor = torch.stack(tensors, dim=0)        # (B, 3, ps, ps)
            # ------------------------------------------------------

            # predict сам делает eval/no_grad/to(settings.DEVICE)
            preds = model.predict(tensor)  # (B, ps, ps) на CPU

            if preds.dim() == 4:
                preds = preds.argmax(dim=1)
            preds = preds.detach().cpu().numpy().astype(np.uint8)

            for (y, x), p in zip(batch_coords, preds):
                patch_names.append(f"{src}_y{y}_x{x}")
                patch_preds.append(p)

        full_pred, coverage = reconstruct_from_patches(
            patch_preds, patch_names, H, W, num_classes, priority_order
        )

        if (coverage == 0).any():
            n_zero = int((coverage == 0).sum())
            raise RuntimeError(
                f"{src}: {n_zero} пикселей без покрытия — сломался "
                f"iter_patch_coords или patch_size/stride"
            )

        cm_total += _confusion_matrix_np(full_pred, gt_idx, num_classes)
        visualizations[src] = (full_pred, gt_idx)

    if verbose:
        print(f"  обработано val-источников: {len(val_source_keys)}")

    metrics = get_all_metrics_from_cm(cm_total, class_names)
    return metrics, visualizations


# ---------------------------------------------------------------------------
# Интервальные оценки
# ---------------------------------------------------------------------------
def get_interval_metrics_from_raw(metrics_df: pd.DataFrame,
                                  confidence_level: float = 0.95) -> pd.DataFrame:
    alpha = 1 - confidence_level
    n = len(metrics_df)
    if n < 2:
        raise ValueError(f"Нужно минимум 2 наблюдения, у нас {n}")
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)

    results = []
    for column in metrics_df.columns:
        data = metrics_df[column]
        mean_val = data.mean()
        sem = data.sem()
        margin = t_critical * sem
        results.append({
            "metric_name": column,
            "lower_bound": mean_val - margin,
            "average": mean_val,
            "upper_bound": mean_val + margin,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def save_visualization(
    full_pred: np.ndarray,
    full_gt: np.ndarray,
    out_path: Path,
    color_map: Dict[int, list],
) -> None:
    """Сохраняет две картинки рядом: pred | gt."""
    def idx_to_rgb(idx):
        out = np.zeros((*idx.shape, 3), dtype=np.uint8)
        for cls, color in color_map.items():
            out[idx == cls] = color
        return out

    pred_rgb = idx_to_rgb(full_pred)
    gt_rgb = idx_to_rgb(full_gt)
    sep = np.full((full_pred.shape[0], 4, 3), 255, dtype=np.uint8)
    combined = np.concatenate([pred_rgb, sep, gt_rgb], axis=1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined).save(out_path)
