# common/metrics_patches.py
import numpy as np
import pandas as pd
from scipy import stats
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# common/metrics_patches.py
# Добавь эти импорты сверху файла:
import albumentations as A
from albumentations.pytorch import ToTensorV2

from common import settings_patches as settings


NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD  = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Confusion matrix и метрики
# ---------------------------------------------------------------------------
def _confusion_matrix_np(preds: np.ndarray, targets: np.ndarray,
                         num_classes: int) -> np.ndarray:
    mask = (targets >= 0) & (targets < num_classes) & \
           (preds >= 0) & (preds < num_classes)
    idx = targets[mask].astype(np.int64) * num_classes + preds[mask].astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def compute_macro_dice_iou(cm):
    num_classes = cm.shape[0]
    dice, iou = [], []
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        dice.append(2 * tp / (2 * tp + fp + fn + 1e-8))
        iou.append(tp / (tp + fp + fn + 1e-8))
    return float(np.mean(dice)), float(np.mean(iou))


def compute_per_class_dice_iou(cm, class_names):
    num_classes = cm.shape[0]
    result = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        name = class_names.get(cls, f'class_{cls}')
        result[f'{name}_dice'] = dice
        result[f'{name}_iou'] = iou
    return result


def compute_per_class_accuracy_precision_recall(cm, class_names):
    num_classes = cm.shape[0]
    total = cm.sum()
    result = {}
    for cls in range(num_classes):
        tp = cm[cls, cls]
        fp = cm[:, cls].sum() - tp
        fn = cm[cls, :].sum() - tp
        tn = total - tp - fp - fn
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        name = class_names.get(cls, f'class_{cls}')
        result[f'{name}_accuracy'] = accuracy
        result[f'{name}_precision'] = precision
        result[f'{name}_recall'] = recall
    return result


def get_all_metrics_from_cm(cm, class_names) -> Dict[str, float]:
    res = compute_per_class_dice_iou(cm, class_names)
    res.update(compute_per_class_accuracy_precision_recall(cm, class_names))
    return res


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
            'metric_name': column,
            'lower_bound': mean_val - margin,
            'average': mean_val,
            'upper_bound': mean_val + margin,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 1) Метрики по патчам
# ---------------------------------------------------------------------------
def evaluate_fold_by_patches(
    model,
    val_loader,
    num_classes: int,
    class_names: Dict[int, str],
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Прогоняет модель по val_loader, накапливает confusion matrix
    по пикселям патчей. Возвращает метрики.
    """
    import torch

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    n_batches = 0

    for images, masks in val_loader:
        preds = model.predict(images)  # (B, H, W) на CPU
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)
        p = preds.detach().cpu().numpy().astype(np.uint8).reshape(-1)
        t = masks.detach().cpu().numpy().astype(np.uint8).reshape(-1)
        cm_total += _confusion_matrix_np(p, t, num_classes)
        n_batches += 1

    if verbose:
        print(f"  батчей в val: {n_batches}, пикселей: {cm_total.sum()}")

    return get_all_metrics_from_cm(cm_total, class_names)


# ---------------------------------------------------------------------------
# 2) Метрики по восстановленным изображениям (без перекрытия — склейка)
# ---------------------------------------------------------------------------
_PATCH_NAME_RE = None


def _iter_patch_coords(H: int, W: int, patch: int, stride: int):
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    for y in ys:
        for x in xs:
            yield y, x


def color_mask_to_index(mask_rgb: np.ndarray, color_map: Dict[int, list]) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in color_map.items():
        m = np.all(mask_rgb == np.array(color, dtype=mask_rgb.dtype), axis=-1)
        out[m] = cls
    return out


def evaluate_fold_reconstructed(
    model,
    val_source_keys: List[str],
    normalized_img_dir: Path,
    normalized_msk_dir: Path,
    num_classes: int,
    color_map: Dict[int, list],
    class_names: Dict[int, str],
    val_extra_transforms,              # <-- принимаем список из модели
    patch_size: int = 256,
    stride: int = 256,
    batch_size: int = 8,
    verbose: bool = True,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Оценка по восстановленным изображениям (без перекрытия — склейка).
    Патчи режутся из normalized_dataset в памяти и прогоняются через
    A.Compose([*val_extra_transforms, ToTensorV2()]) — тот же transform,
    что и в val_loader.
    """
    import torch

    assert stride == patch_size, (
        f"Эта функция без перекрытия: stride ({stride}) должен "
        f"равняться patch_size ({patch_size})"
    )

    # Собираем transform ровно как в _make_val_loader
    val_transform = A.Compose([*val_extra_transforms, ToTensorV2()])

    cm_total = np.zeros((num_classes, num_classes), dtype=np.int64)
    visualizations: Dict[str, np.ndarray] = {}

    for src in val_source_keys:
        img_path = normalized_img_dir / f"{src}.png"
        msk_path = normalized_msk_dir / f"{src}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Нет изображения {img_path}")
        if not msk_path.exists():
            raise FileNotFoundError(f"Нет маски {msk_path}")

        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        gt_rgb = np.array(Image.open(msk_path).convert("RGB"))
        H, W = img_rgb.shape[:2]
        assert gt_rgb.shape[:2] == (H, W), \
            f"{src}: изображение {img_rgb.shape[:2]} != маска {gt_rgb.shape[:2]}"

        gt_idx = color_mask_to_index(gt_rgb, color_map)

        full_pred = np.zeros((H, W), dtype=np.uint8)
        coords = list(_iter_patch_coords(H, W, patch_size, stride))

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            batch_imgs = [img_rgb[y:y + patch_size, x:x + patch_size]
                          for (y, x) in batch_coords]
            batch_arr = np.stack(batch_imgs, axis=0)  # (B, ps, ps, 3) uint8

            # ---- прогон через тот же transform, что в val_loader ----
            tensors = []
            for b in range(batch_arr.shape[0]):
                out = val_transform(image=batch_arr[b])
                tensors.append(out["image"])           # (3, ps, ps) float32
            tensor = torch.stack(tensors, dim=0)       # (B, 3, ps, ps)
            # ---------------------------------------------------------

            preds = model.predict(tensor)              # (B, ps, ps) CPU
            if preds.dim() == 4:
                preds = preds.argmax(dim=1)
            preds = preds.detach().cpu().numpy().astype(np.uint8)

            for (y, x_), p in zip(batch_coords, preds):
                full_pred[y:y + patch_size, x_:x_ + patch_size] = p

        cm_total += _confusion_matrix_np(full_pred, gt_idx, num_classes)
        visualizations[src] = (full_pred, gt_idx)

    if verbose:
        print(f"  обработано val-источников: {len(val_source_keys)}")

    return get_all_metrics_from_cm(cm_total, class_names), visualizations

# ---------------------------------------------------------------------------
# Визуализация
# ---------------------------------------------------------------------------
def save_visualization(
    full_pred: np.ndarray,
    full_gt: np.ndarray,
    out_path: Path,
    color_map: Dict[int, list],
) -> None:
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
