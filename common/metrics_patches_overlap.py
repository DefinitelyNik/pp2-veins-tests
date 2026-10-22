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
from common.experiment_utils_overlap import strict_rgb_to_labels, read_rgb


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
    return strict_rgb_to_labels(mask_rgb, color_map)


# ---------------------------------------------------------------------------
# Нарезка изображения на патчи прямо в памяти
# ---------------------------------------------------------------------------
def iter_patch_coords(H: int, W: int, patch: int, stride: int):
    """
    Генерирует координаты (y, x) верхнего левого угла патчей,
    покрывающих изображение HxW. Логика та же, что в скрипте нарезки:
    последний патч добивается до края, если сетка не попадает точно.
    """
    if H < 1 or W < 1 or patch < 1 or not 0 < stride <= patch:
        raise ValueError("Require positive image sizes and 0 < stride <= patch.")
    def starts(length):
        if length <= patch:
            return [0]
        result = list(range(0, length - patch + 1, stride))
        if result[-1] != length - patch:
            result.append(length - patch)
        return result
    for y in starts(H):
        for x in starts(W):
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
    if sorted(priority_order) != list(range(num_classes)):
        raise ValueError("priority_order must contain each class exactly once.")
    if len(patch_preds) != len(patch_names):
        raise ValueError("Different numbers of predictions and patch names.")

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

    full_pred[coverage == 0] = 0
    return full_pred, coverage


# ---------------------------------------------------------------------------
# Confusion matrix и метрики
# ---------------------------------------------------------------------------
def _confusion_matrix_np(preds: np.ndarray, targets: np.ndarray,
                         num_classes: int) -> np.ndarray:
    """Быстрая confusion matrix без sklearn."""
    if preds.shape != targets.shape:
        raise ValueError("Prediction/target shape mismatch.")
    if not np.issubdtype(preds.dtype, np.integer) or not np.issubdtype(targets.dtype, np.integer):
        raise ValueError("Confusion matrix inputs must be integer label maps.")
    mask = (targets >= 0) & (targets < num_classes) & \
           (preds >= 0) & (preds < num_classes)
    if not mask.all():
        raise ValueError("Invalid labels: refusing to silently discard pixels.")
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
    macro_dice, macro_iou = compute_macro_dice_iou(cm)
    per_class["macro_dice_all"] = macro_dice
    per_class["macro_iou_all"] = macro_iou
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    per_class["foreground_macro_dice"] = float(dice[1:].mean())
    per_class["foreground_macro_iou"] = float(iou[1:].mean())
    vessel_tp = float(cm[1:, 1:].sum())
    vessel_fp, vessel_fn = float(cm[0, 1:].sum()), float(cm[1:, 0].sum())
    per_class["vessel_union_dice"] = 2 * vessel_tp / (2 * vessel_tp + vessel_fp + vessel_fn + 1e-8)
    per_class["vessel_union_iou"] = vessel_tp / (vessel_tp + vessel_fp + vessel_fn + 1e-8)
    return per_class


# ---------------------------------------------------------------------------
# Оценка фолда по восстановленным изображениям
# ---------------------------------------------------------------------------
def make_blending_window(patch_size):
    axis = np.hanning(patch_size).astype(np.float32)
    return np.maximum(np.outer(axis, axis), 1e-3).astype(np.float32)


def binary_topology_metrics(pred, target):
    """Hard metrics, not the differentiable training surrogate.

    8-connectivity; HD95 = percentile of pooled bidirectional surface
    distances in pixels at evaluated resolution. No small-component removal.
    Empty/empty: clDice=1, HD95=0; exactly one empty: clDice=0, HD95=NaN.
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    pred, target = pred.astype(bool), target.astype(bool)
    structure = np.ones((3, 3), dtype=bool)
    components_pred = int(ndi.label(pred, structure)[1])
    components_gt = int(ndi.label(target, structure)[1])
    result = {"components_pred": components_pred, "components_gt": components_gt}
    if not pred.any() or not target.any():
        both_empty = not pred.any() and not target.any()
        result.update(cldice=float(both_empty), hd95_px=0.0 if both_empty else np.nan)
        return result
    skel_pred, skel_gt = skeletonize(pred), skeletonize(target)
    precision = float(target[skel_pred].mean())
    recall = float(pred[skel_gt].mean())
    result["cldice"] = 2 * precision * recall / (precision + recall + 1e-8)
    surface_pred = pred & ~ndi.binary_erosion(pred, structure, border_value=0)
    surface_gt = target & ~ndi.binary_erosion(target, structure, border_value=0)
    d_gt = ndi.distance_transform_edt(~surface_gt)
    distances_pred = d_gt[surface_pred]
    del d_gt
    d_pred = ndi.distance_transform_edt(~surface_pred)
    distances_gt = d_pred[surface_gt]
    result["hd95_px"] = float(np.percentile(
        np.concatenate([distances_pred, distances_gt]), 95
    ))
    return result


def evaluate_fold_overlap(
    model, val_source_keys, normalized_img_dir, normalized_msk_dir,
    num_classes, color_map, class_names, val_extra_transforms,
    patch_size=256, stride=128, priority_order=None, batch_size=8,
    verbose=True, tta=False, per_image_records=None, visualization_dir=None,
    fold_num=None, keep_visualizations=True, compute_topology=True,
    patient_groups=None, confusion_matrix_path=None,
):
    """Probability blending; optional streaming visualizations and per-image CSV data.

    priority_order is retained for signature compatibility but is not applied.
    Returns (pixel-pooled fold metrics, optional visualizations). Topology is
    computed per image and saved into per_image_records, not pixel-pooled.
    """
    import torch
    import warnings
    if priority_order is not None:
        warnings.warn("priority_order ignored: probability blending has no class priority.")
    if not val_source_keys or len(set(val_source_keys)) != len(val_source_keys):
        raise ValueError("Validation sources must be nonempty and unique.")
    if batch_size < 1 or patch_size % 32 or not 0 < stride <= patch_size:
        raise ValueError("Invalid inference patch, stride or batch size.")
    if not hasattr(model, "predict_proba"):
        raise TypeError("Probability blending requires model.predict_proba().")
    normalized_img_dir, normalized_msk_dir = Path(normalized_img_dir), Path(normalized_msk_dir)
    transform = A.Compose([*(val_extra_transforms or []), ToTensorV2()])
    window = make_blending_window(patch_size)
    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    visualizations = {}
    for source in val_source_keys:
        rgb = read_rgb(normalized_img_dir / f"{source}.png")
        gt_rgb = read_rgb(normalized_msk_dir / f"{source}.png")
        if rgb.shape != gt_rgb.shape:
            raise ValueError(f"Source image/mask dimensions differ: {source}")
        gt = color_mask_to_index(gt_rgb, color_map)
        h, w = rgb.shape[:2]
        padded = np.pad(rgb, ((0, max(0, patch_size-h)),
                             (0, max(0, patch_size-w)), (0, 0)), mode="edge")
        ph, pw = padded.shape[:2]
        coords = list(iter_patch_coords(ph, pw, patch_size, stride))
        probability_sum = np.zeros((num_classes, ph, pw), dtype=np.float32)
        weight_sum = np.zeros((ph, pw), dtype=np.float32)
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start:start + batch_size]
            tensors = [transform(image=padded[y:y+patch_size, x:x+patch_size])["image"]
                       for y, x in batch_coords]
            tensor = torch.stack(tensors).float()
            probs = model.predict_proba(tensor, tta=tta).detach().cpu().numpy()
            expected = (len(batch_coords), num_classes, patch_size, patch_size)
            if probs.shape != expected or not np.isfinite(probs).all():
                raise ValueError(f"Invalid probabilities: shape={probs.shape}, expected={expected}")
            if np.any(probs < -1e-5) or np.any(probs > 1 + 1e-5) or not np.allclose(probs.sum(1), 1, atol=1e-4):
                raise ValueError("predict_proba must return normalized softmax probabilities.")
            for (y, x), patch_probs in zip(batch_coords, probs):
                probability_sum[:, y:y+patch_size, x:x+patch_size] += patch_probs * window[None]
                weight_sum[y:y+patch_size, x:x+patch_size] += window
        if np.any(weight_sum <= 0):
            raise RuntimeError(f"Uncovered image pixels: {source}")
        probability_sum /= weight_sum[None]
        prediction = probability_sum[:, :h, :w].argmax(0).astype(np.uint8)
        del probability_sum, weight_sum, padded
        cm = _confusion_matrix_np(prediction, gt, num_classes)
        cm_total += cm
        row = {"source_key": source, "fold": fold_num,
               "patient_id": (patient_groups or {}).get(source, source),
               "height": h, "width": w}
        row.update(get_all_metrics_from_cm(cm, class_names))
        if compute_topology:
            for name, pred_binary, target_binary in (
                ("vessel_union", prediction > 0, gt > 0),
                ("arteries", prediction == 1, gt == 1),
                ("veins", prediction == 2, gt == 2),
            ):
                row.update({f"{name}_{key}": value for key, value in
                            binary_topology_metrics(pred_binary, target_binary).items()})
        if per_image_records is not None:
            per_image_records.append(row)
        if keep_visualizations:
            visualizations[source] = (prediction, gt)
        if visualization_dir is not None:
            prefix = f"fold{fold_num}_" if fold_num is not None else ""
            save_visualization(prediction, gt, Path(visualization_dir) / f"{prefix}{source}.png", color_map)
        if verbose:
            print(f"[eval] {source}: {len(coords)} patches, foreground Dice={row['foreground_macro_dice']:.4f}")
    if confusion_matrix_path is not None:
        path = Path(confusion_matrix_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, cm_total)
    return get_all_metrics_from_cm(cm_total, class_names), visualizations


# ---------------------------------------------------------------------------
# Интервальные оценки
# ---------------------------------------------------------------------------
def get_interval_metrics_from_raw(metrics_df: pd.DataFrame,
                                  confidence_level: float = 0.95) -> pd.DataFrame:
    import warnings
    warnings.warn("Legacy t-interval over folds: descriptive only, folds are not independent. "
                  "Use bootstrap_per_image for conditional image/group uncertainty.")
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


def bootstrap_per_image(frame, confidence_level=0.95, samples=2000, seed=42):
    """Percentile cluster bootstrap of mean per-image OOF metrics.

    Groups are patient IDs if supplied, otherwise source image IDs. Conditional
    on these trained models/splits; does NOT measure training-seed uncertainty.
    NaNs (undefined HD95 for one-empty masks) are excluded per metric and counted.
    """
    if not 0 < confidence_level < 1 or samples < 1 or frame.empty:
        raise ValueError("Invalid bootstrap parameters or empty input.")
    excluded = {"fold", "height", "width", "source_key", "patient_id"}
    columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    values = frame[columns].to_numpy(dtype=np.float64)
    identifiers = frame["patient_id"].astype(str).to_numpy()
    unique = sorted(set(identifiers))
    groups = [np.flatnonzero(identifiers == identifier) for identifier in unique]
    rng = np.random.default_rng(seed)
    def means(array):
        finite = np.isfinite(array)
        counts = finite.sum(0)
        sums = np.where(finite, array, 0).sum(0)
        return np.divide(sums, counts, out=np.full(len(columns), np.nan), where=counts > 0)
    estimates = np.empty((samples, len(columns)), dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[g] for g in chosen])
        estimates[index] = means(values[rows])
    alpha = (1 - confidence_level) / 2
    average = means(values)
    results = []
    for index, column in enumerate(columns):
        valid = estimates[:, index][np.isfinite(estimates[:, index])]
        low, high = np.quantile(valid, [alpha, 1-alpha]) if len(valid) else (np.nan, np.nan)
        results.append({
            "metric_name": column, "average": average[index],
            "lower_bound": low, "upper_bound": high,
            "valid_images": int(np.isfinite(values[:, index]).sum()),
            "undefined_images": int((~np.isfinite(values[:, index])).sum()),
            "groups": len(groups), "bootstrap_samples": samples,
            "scope": "conditional_mean_per_image_oof_cluster_bootstrap",
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
