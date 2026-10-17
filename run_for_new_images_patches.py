# run_for_new_images_patches.py
import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from common import settings_patches as settings
from common import metrics_patches as metrics

# --- ВЫБОР МОДЕЛИ ---
from UNETPP_WCD.model_patches import (
    ImprovedUNetPlusPlus, val_extra_transforms
)
ModelClass = ImprovedUNetPlusPlus


# --- ПУТИ ---
NORMALIZED_EXTERNAL_ROOT = Path("normalized_external_datasets")
OUTPUT_ROOT = Path("external_eval_patches")
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


# ---------- Обработка одного датасета для одной модели ----------
def evaluate_dataset_with_model(
    model,
    dataset_dir: Path,
    dataset_name: str,
    val_transform,
    patch_size: int,
    stride: int,
    batch_size: int,
    vis_dir: Path | None,
    verbose: bool = True,
):
    """
    Режет каждое изображение из normalized_external_datasets/<name>/images/ на
    патчи patch_size со stride, прогоняет через модель и склеивает обратно.
    Считает метрики ТОЛЬКО по восстановленным изображениям.

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
        full_pred = np.zeros((H, W), dtype=np.uint8)

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

            for (y, x), p in zip(batch_coords, preds):
                full_pred[y:y + patch_size, x:x + patch_size] = p

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
    parser.add_argument("--no-vis", action="store_true",
                        help="Отключить визуализации.")
    args = parser.parse_args()

    assert args.stride == args.patch_size, \
        "Скрипт для нарезки без перекрытия: stride должен быть равен patch_size"

    default_datasets = ["RAV", "LES-AV", "CHASE-AV", "DRIVE_AV"]
    datasets = args.datasets if args.datasets else default_datasets

    dataset_dirs = [NORMALIZED_EXTERNAL_ROOT / name for name in datasets]

    print(f"Модель: {MODEL_NAME}")
    print(f"Результаты: {MODEL_OUTPUT_ROOT}")
    print(f"patch_size={args.patch_size}, stride={args.stride}, "
          f"batch_size={args.batch_size}")
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

        # сводный CSV по фолдам для этого датасета
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
