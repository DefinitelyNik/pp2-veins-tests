# run_for_new_images_patches_overlap.py
import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from common import settings_patches_overlap as settings
from common import metrics_patches_overlap as metrics

# --- ВЫБОР МОДЕЛИ ---
from UNETPP_WCD.model_patches_overlap import (
    ImprovedUNetPlusPlus, val_extra_transforms
)
ModelClass = ImprovedUNetPlusPlus


# --- ПУТИ ---
NORMALIZED_EXTERNAL_ROOT = Path("normalized_external_datasets")
OUTPUT_ROOT = Path("external_eval_patches_overlap")
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

MODEL_NAME = ModelClass.get_model_name()
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / MODEL_NAME
MODEL_OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)


# ---------- Утилиты ----------
def iter_patch_coords(H: int, W: int, patch: int, stride: int):
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    for y in ys:
        for x in xs:
            yield y, x


def color_mask_to_index(mask_rgb: np.ndarray, color_map) -> np.ndarray:
    h, w, _ = mask_rgb.shape
    out = np.zeros((h, w), dtype=np.uint8)
    for cls, color in color_map.items():
        m = np.all(mask_rgb == np.array(color, dtype=mask_rgb.dtype), axis=-1)
        out[m] = cls
    return out


def mask_to_rgb(mask_np: np.ndarray, color_map) -> np.ndarray:
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in color_map.items():
        rgb[mask_np == cls] = np.array(color, dtype=np.uint8)
    return rgb


def make_overlay(image_rgb: np.ndarray, mask_np: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    overlay = image_rgb.copy().astype(np.float32)
    color_mask = mask_to_rgb(mask_np, settings.COLOR_MAP).astype(np.float32)
    non_bg = (mask_np != 0)[..., None]
    blended = overlay * (1 - alpha * non_bg) + color_mask * (alpha * non_bg)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_val_transform():
    """Тот же A.Compose, что и в val_loader."""
    transforms = list(val_extra_transforms) if val_extra_transforms else []
    transforms.append(ToTensorV2())
    return A.Compose(transforms)


def apply_transform_to_batch(transform, batch_uint8: np.ndarray) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float32 через тот же transform, что в val_loader."""
    tensors = []
    for i in range(batch_uint8.shape[0]):
        out = transform(image=batch_uint8[i])
        tensors.append(out["image"])
    return torch.stack(tensors, dim=0)


def collect_image_mask_pairs(dataset_dir: Path):
    images_dir = dataset_dir / "images"
    masks_dir = dataset_dir / "masks"
    assert images_dir.is_dir(), f"Нет папки {images_dir}"
    assert masks_dir.is_dir(), f"Нет папки {masks_dir}"

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def collect(d):
        paths = []
        for ext in exts:
            paths.extend(d.glob(f"*{ext}"))
        return sorted(paths)

    image_paths = collect(images_dir)
    mask_paths = collect(masks_dir)

    masks_by_stem = {p.stem: p for p in mask_paths}
    pairs = []
    missing = []
    for img_path in image_paths:
        m = masks_by_stem.get(img_path.stem)
        if m is None:
            missing.append(img_path.name)
            continue
        pairs.append((img_path, m))

    if missing:
        raise FileNotFoundError(
            f"Нет масок для {len(missing)} картинок. Примеры: {missing[:5]}"
        )
    return pairs


# ---------- Восстановление из overlap-патчей ----------
def reconstruct_from_patches(
    patch_preds: list,
    coords: list,
    full_h: int,
    full_w: int,
    num_classes: int,
    priority_order: list,
) -> np.ndarray:
    """
    Собирает полное предсказание из overlap-патчей голосованием.
    При равенстве голосов — приоритет по priority_order
    (первый в списке — самый приоритетный).

    patch_preds : список (ph, pw) uint8
    coords      : список (y, x) тех же патчей
    """
    votes = np.zeros((num_classes, full_h, full_w), dtype=np.int32)

    for pred, (y, x) in zip(patch_preds, coords):
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

    # Перестановка каналов в порядке приоритета: argmax вернёт индекс
    # в переставленном массиве, который соответствует priority_order[idx].
    votes_perm = votes[priority_order, :, :]
    idx = np.argmax(votes_perm, axis=0)
    full_pred = np.array(priority_order, dtype=np.uint8)[idx]
    return full_pred


# ---------- Обработка одного датасета для одной модели ----------
def evaluate_dataset_with_model(
    model,
    dataset_dir: Path,
    dataset_name: str,
    val_transform,
    patch_size: int,
    stride: int,
    batch_size: int,
    priority_order: list,
    vis_dir: Path | None,
    verbose: bool = True,
):
    """
    Для каждого изображения из normalized_external_datasets/<name>/images/:
      - режет на overlap-патчи,
      - прогоняет через модель,
      - восстанавливает полное предсказание голосованием,
      - считает метрики по восстановленному изображению.
    Возвращает (per_image_rows, agg_metrics_recon).
    """
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    cm_recon_total = np.zeros((settings.NUM_CLASSES, settings.NUM_CLASSES), dtype=np.int64)

    pairs = collect_image_mask_pairs(dataset_dir)

    for img_path, msk_path in pairs:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        msk_rgb = np.array(Image.open(msk_path).convert("RGB"))
        H, W = img_rgb.shape[:2]
        assert msk_rgb.shape[:2] == (H, W), \
            f"{img_path.name}: изображение {img_rgb.shape[:2]} != маска {msk_rgb.shape[:2]}"

        gt_idx = color_mask_to_index(msk_rgb, settings.COLOR_MAP)

        coords = list(iter_patch_coords(H, W, patch_size, stride))
        patch_preds = []

        for i in range(0, len(coords), batch_size):
            batch_coords = coords[i:i + batch_size]
            batch_imgs = [img_rgb[y:y + patch_size, x:x + patch_size]
                          for (y, x) in batch_coords]
            batch_arr = np.stack(batch_imgs, axis=0)

            tensor = apply_transform_to_batch(val_transform, batch_arr)
            preds = model.predict(tensor)
            if preds.dim() == 4:
                preds = preds.argmax(dim=1)
            preds = preds.detach().cpu().numpy().astype(np.uint8)

            for p in preds:
                patch_preds.append(p)

        full_pred = reconstruct_from_patches(
            patch_preds, coords, H, W,
            settings.NUM_CLASSES, priority_order,
        )

        # confusion matrix по восстановленному изображению
        cm_img = metrics._confusion_matrix_np(
            full_pred.reshape(-1), gt_idx.reshape(-1), settings.NUM_CLASSES
        )
        cm_recon_total += cm_img

        # per-image метрики
        m = metrics.get_all_metrics_from_cm(cm_img, settings.CLASS_NAMES)
        m["image_path"] = f"{dataset_name}/{img_path.name}"
        per_image_rows.append(m)

        # визуализации
        if vis_dir is not None:
            stem = img_path.stem
            Image.fromarray(mask_to_rgb(gt_idx, settings.COLOR_MAP)).save(
                vis_dir / f"{stem}_gt.png"
            )
            Image.fromarray(mask_to_rgb(full_pred, settings.COLOR_MAP)).save(
                vis_dir / f"{stem}_pred.png"
            )
            Image.fromarray(make_overlay(img_rgb, full_pred)).save(
                vis_dir / f"{stem}_overlay.png"
            )

    agg_recon = metrics.get_all_metrics_from_cm(cm_recon_total, settings.CLASS_NAMES)

    if verbose:
        print(f"    изображений обработано: {len(pairs)}")

    return per_image_rows, agg_recon


# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Список датасетов внутри normalized_external_datasets/. "
             "По умолчанию: RAV LES-AV CHASE-AV DRIVE_AV",
    )
    parser.add_argument("--batch-size", type=int, default=settings.BATCH_SIZE)
    parser.add_argument("--patch-size", type=int, default=settings.PATCH_SIZE)
    parser.add_argument("--stride", type=int, default=settings.STRIDE)
    parser.add_argument(
        "--priority", nargs="*", type=int, default=[3, 1, 2, 4, 0],
        help="Порядок приоритета при равенстве голосов. "
             "По умолчанию: 3 1 2 4 0 (пересечения > артерии > вены > каппиляры > фон).",
    )
    parser.add_argument("--no-vis", action="store_true",
                        help="Отключить визуализации.")
    args = parser.parse_args()

    assert args.stride < args.patch_size, (
        f"Для overlap stride ({args.stride}) должен быть меньше "
        f"patch_size ({args.patch_size})"
    )
    assert set(args.priority) == set(range(settings.NUM_CLASSES)), (
        f"priority должен содержать все {settings.NUM_CLASSES} классов, "
        f"получено: {args.priority}"
    )

    default_datasets = ["RAV", "LES-AV", "CHASE-AV", "DRIVE_AV"]
    datasets = args.datasets if args.datasets else default_datasets

    dataset_dirs = [NORMALIZED_EXTERNAL_ROOT / name for name in datasets]

    print(f"Модель: {MODEL_NAME}")
    print(f"Результаты: {MODEL_OUTPUT_ROOT}")
    print(f"patch_size={args.patch_size}, stride={args.stride}, "
          f"batch_size={args.batch_size}, priority={args.priority}")
    print("Датасеты:")
    for d in dataset_dirs:
        print("  -", d.name)

    val_transform = build_val_transform()

    # --- Загрузка моделей всех фолдов ---
    models = []
    for fold in range(1, settings.FOLDS_NUM + 1):
        model = ModelClass()
        path = ModelClass.get_model_save_path(fold)
        print(f"Загрузка модели фолда {fold} из {path}")
        model.load(str(path))
        models.append(model)

    # --- По каждому датасету ---
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        if not dataset_dir.is_dir():
            print(f"[SKIP] нет папки {dataset_dir}")
            continue

        out_dir = MODEL_OUTPUT_ROOT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## Датасет: {dataset_name} ##########")
        print(f"Результаты: {out_dir}")

        all_aggs_recon = []

        for fold_idx, model in enumerate(models, start=1):
            print(f"\n=== Фолд {fold_idx} ===")

            vis_dir = None if args.no_vis else out_dir / "visualizations" / f"fold_{fold_idx}"

            per_image_rows, agg_recon = evaluate_dataset_with_model(
                model=model,
                dataset_dir=dataset_dir,
                dataset_name=dataset_name,
                val_transform=val_transform,
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
                priority_order=args.priority,
                vis_dir=vis_dir,
            )

            # per-image CSV
            per_image_df = pd.DataFrame(per_image_rows)
            meta_cols = ["image_path"]
            other = [c for c in per_image_df.columns if c not in meta_cols]
            per_image_df = per_image_df[meta_cols + other]
            per_image_path = out_dir / f"per_image_fold_{fold_idx}.csv"
            per_image_df.to_csv(per_image_path, index=False)

            # агрегат по фолду
            agg_recon["fold"] = fold_idx
            all_aggs_recon.append(agg_recon)

            print(f"  per-image метрики: {per_image_path}")
            if vis_dir is not None:
                print(f"  визуализации:      {vis_dir}")

        # сводный CSV по фолдам для датасета
        agg_df = pd.DataFrame(all_aggs_recon)
        meta_cols = ["fold"]
        other = [c for c in agg_df.columns if c not in meta_cols]
        agg_df = agg_df[meta_cols + other]
        agg_path = out_dir / "aggregate_reconstructed_all_folds.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"\nСводный агрегат по датасету: {agg_path}")

    # --- Сводный файл по всем датасетам ---
    rows = []
    for dataset_dir in dataset_dirs:
        path = MODEL_OUTPUT_ROOT / dataset_dir.name / "aggregate_reconstructed_all_folds.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.insert(0, "dataset", dataset_dir.name)
            rows.append(df)

    if rows:
        combined = pd.concat(rows, ignore_index=True)
        meta_cols = ["dataset", "fold"]
        other = [c for c in combined.columns if c not in meta_cols]
        combined = combined[meta_cols + other]
        combined_path = MODEL_OUTPUT_ROOT / "aggregate_all_datasets_reconstructed.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nСводка по всем датасетам: {combined_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
