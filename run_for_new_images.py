# eval_external.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from common import settings
from common import metrics


# --- ВЫБОР МОДЕЛИ: замени на свою, как в основном пайплайне ---
<<<<<<< HEAD
from UNETPP_WCT.model import UNetPPWCT, val_extra_transforms
ModelClass = UNetPPWCT
=======
from UNETPP_WCD.model import ImprovedUNetPlusPlus, val_extra_transforms
ModelClass = ImprovedUNetPlusPlus
>>>>>>> d2abc896 (feat pp-veins: add patching)


# --- ПУТИ ---
EXTERNAL_ROOT = settings.BASE_DIR / "external_datasets"
OUTPUT_ROOT = settings.BASE_DIR / "external_eval"
OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)

MODEL_NAME = ModelClass.get_model_name()
MODEL_OUTPUT_ROOT = OUTPUT_ROOT / MODEL_NAME
MODEL_OUTPUT_ROOT.mkdir(exist_ok=True, parents=True)


# Нормализация (совпадает с той, что в модели)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ExternalSegmentationDataset(Dataset):
    """
    Возвращает (image_tensor, mask_tensor, rel_path, raw_image_rgb).

    image_tensor : torch.Tensor CHW float  — нормализованный вход для модели
    mask_tensor  : torch.Tensor HW   long   — индексы классов
    rel_path     : str                       — "<dataset_name>/<image_name>"
    raw_image_rgb: np.ndarray HWC uint8      — картинка после Resize, до Normalize,
                                               для overlay-визуализации
    """

    def __init__(self, dataset_root: Path, transform=None):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_root.name
        self.transform = transform

        images_dir = dataset_root / "images"
        masks_dir = dataset_root / "masks"

        if not images_dir.exists():
            raise FileNotFoundError(f"Нет папки {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Нет папки {masks_dir}")

        exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")

        def collect(directory):
            paths = []
            for ext in exts:
                paths.extend(directory.glob(ext))
            return sorted(paths)

        image_paths = collect(images_dir)
        mask_paths = collect(masks_dir)

        print(f"  [{self.dataset_name}] images: {len(image_paths)}, masks: {len(mask_paths)}")
        if not image_paths:
            raise FileNotFoundError(f"В {images_dir} нет картинок")
        if not mask_paths:
            raise FileNotFoundError(f"В {masks_dir} нет масок")

        # Маски ищем по stem (имя без расширения) — картинки .jpg, маски .png, это ок
        masks_by_stem = {}
        for p in mask_paths:
            masks_by_stem.setdefault(p.stem, p)

        pairs = []
        missing = []
        for img_path in image_paths:
            mask_path = masks_by_stem.get(img_path.stem)
            if mask_path is None:
                missing.append(img_path.name)
                continue
            pairs.append((img_path, mask_path))

        if missing:
            raise FileNotFoundError(
                f"Для {len(missing)} картинок нет масок с тем же stem. "
                f"Примеры: {missing[:5]}"
            )
        if not pairs:
            raise FileNotFoundError(
                f"Не удалось сопоставить ни одной пары в {dataset_root}"
            )

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def rgb_to_labels(self, mask_rgb: np.ndarray) -> np.ndarray:
        h, w, _ = mask_rgb.shape
        mask_labels = np.zeros((h, w), dtype=np.uint8)
        for class_idx, color in settings.COLOR_MAP.items():
            color_mask = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
            mask_labels[color_mask] = class_idx
        return mask_labels

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("RGB"))
        mask = self.rgb_to_labels(mask)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image_t = augmented["image"]         # tensor CHW, normalised
            mask_t = augmented["mask"].long()    # tensor HW

            # Денормализуем для overlay: raw = norm * std + mean
            raw = image_t.cpu().numpy().transpose(1, 2, 0)
            raw = raw * np.array(STD) + np.array(MEAN)
            raw = np.clip(raw, 0, 1)
            raw_uint8 = (raw * 255).astype(np.uint8)
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask_t = torch.from_numpy(mask).long()
            raw_uint8 = image.astype(np.uint8)

        rel_path = f"{self.dataset_name}/{img_path.name}"
        return image_t, mask_t, rel_path, raw_uint8


def build_loader_for_dataset(dataset_root: Path, batch_size: int = None) -> DataLoader:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    transforms = []
    if val_extra_transforms:
        transforms.extend(val_extra_transforms)
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)

    ds = ExternalSegmentationDataset(dataset_root, transform=transform)
    bs = batch_size or settings.BATCH_SIZE
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=settings.NUM_WORKERS,
        pin_memory=True,
    )


def mask_to_rgb(mask_np: np.ndarray) -> np.ndarray:
    """mask_np: HxW uint8 с индексами классов -> HxWx3 uint8 RGB."""
    h, w = mask_np.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in settings.COLOR_MAP.items():
        rgb[mask_np == cls] = np.array(color, dtype=np.uint8)
    return rgb


def make_overlay(image_rgb: np.ndarray, mask_np: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    image_rgb : HxWx3 uint8 — исходная картинка
    mask_np   : HxW   uint8 — индексы классов
    Возвращает HxWx3 uint8 — наложение маски на картинку с прозрачностью alpha.
    Класс 0 (background) не накладывается.
    """
    overlay = image_rgb.copy().astype(np.float32)
    color_mask = mask_to_rgb(mask_np).astype(np.float32)
    non_bg = (mask_np != 0)[..., None]  # HxWx1
    blended = overlay * (1 - alpha * non_bg) + color_mask * (alpha * non_bg)
    return np.clip(blended, 0, 255).astype(np.uint8)


def evaluate_fold(model, loader, num_classes, class_names, vis_dir: Path):
    """
    Прогоняет одну модель по всему датасету.
    Сохраняет визуализации в vis_dir (если vis_dir is not None).
    Возвращает (per_image_rows, agg_metrics).
    """
    if vis_dir is not None:
        vis_dir.mkdir(parents=True, exist_ok=True)

    per_image_rows = []
    all_preds = []
    all_targets = []

    for images, masks, rel_paths, raw_images in loader:
        preds = model.predict(images)  # BxHxW, cpu, long

        for b in range(images.size(0)):
            pred_np = preds[b].cpu().numpy().astype(np.uint8)
            target_np = masks[b].cpu().numpy().astype(np.uint8)

            raw_np = raw_images[b]
            if isinstance(raw_np, torch.Tensor):
                raw_np = raw_np.cpu().numpy()
            raw_np = raw_np.astype(np.uint8)

            # --- метрики по одной картинке ---
            pred_flat = pred_np.reshape(-1)
            target_flat = target_np.reshape(-1)
            m = metrics.get_all_metrics(
                pred_flat, target_flat,
                num_classes=num_classes,
                class_names=class_names,
            )
            m["image_path"] = rel_paths[b]
            per_image_rows.append(m)

            # --- визуализации ---
            if vis_dir is not None:
                stem = Path(rel_paths[b]).stem
                Image.fromarray(mask_to_rgb(target_np)).save(vis_dir / f"{stem}_gt.png")
                Image.fromarray(mask_to_rgb(pred_np)).save(vis_dir / f"{stem}_pred.png")
                Image.fromarray(make_overlay(raw_np, pred_np)).save(vis_dir / f"{stem}_overlay.png")

            all_preds.append(pred_flat)
            all_targets.append(target_flat)

    # --- агрегат по всему датасету (один вызов на объединённых пикселях) ---
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    agg_metrics = metrics.get_all_metrics(
        all_preds, all_targets,
        num_classes=num_classes,
        class_names=class_names,
    )
    return per_image_rows, agg_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Список имён датасетов внутри external_datasets/. "
             "Если не указано — берутся все подпапки.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true",
                        help="Отключить визуализации (быстрее).")
    args = parser.parse_args()

    if args.datasets:
        dataset_dirs = [EXTERNAL_ROOT / name for name in args.datasets]
    else:
        dataset_dirs = [p for p in sorted(EXTERNAL_ROOT.iterdir()) if p.is_dir()]

    if not dataset_dirs:
        raise FileNotFoundError(f"В {EXTERNAL_ROOT} нет подпапок с датасетами")

    print(f"Модель: {MODEL_NAME}")
    print(f"Результаты: {MODEL_OUTPUT_ROOT}")
    print("Будут обработаны датасеты:")
    for d in dataset_dirs:
        print("  -", d.name)

    # --- Загружаем модели всех фолдов ---
    models = []
    for fold in range(1, settings.FOLDS_NUM + 1):
        model = ModelClass()
        path = ModelClass.get_model_save_path(fold)
        print(f"Загрузка модели фолда {fold} из {path}")
        model.load(str(path))
        models.append(model)

    # --- Для каждого датасета — своя папка внутри MODEL_OUTPUT_ROOT ---
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name
        out_dir = MODEL_OUTPUT_ROOT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n########## Датасет: {dataset_name} ##########")
        print(f"Результаты: {out_dir}")

        loader = build_loader_for_dataset(dataset_dir, batch_size=args.batch_size)

        all_aggregates = []

        for fold_idx, model in enumerate(models, start=1):
            print(f"\n=== Фолд {fold_idx} ===")

            vis_dir = out_dir / "visualizations" / f"fold_{fold_idx}"
            if args.no_vis:
                vis_dir = None

            per_image_rows, agg_metrics = evaluate_fold(
                model, loader,
                num_classes=settings.NUM_CLASSES,
                class_names=settings.CLASS_NAMES,
                vis_dir=vis_dir,
            )

            # per-image CSV
            per_image_df = pd.DataFrame(per_image_rows)
            meta_cols = ["image_path"]
            other = [c for c in per_image_df.columns if c not in meta_cols]
            per_image_df = per_image_df[meta_cols + other]
            per_image_path = out_dir / f"per_image_fold_{fold_idx}.csv"
            per_image_df.to_csv(per_image_path, index=False)

            # агрегат по всему датасету для этого фолда
            agg_metrics["fold"] = fold_idx
            all_aggregates.append(agg_metrics)

            print(f"  per-image метрики: {per_image_path}")
            if vis_dir is not None:
                print(f"  визуализации:      {vis_dir}")

        # сводный агрегат по фолдам для этого датасета
        agg_df = pd.DataFrame(all_aggregates)
        meta_cols = ["fold"]
        other = [c for c in agg_df.columns if c not in meta_cols]
        agg_df = agg_df[meta_cols + other]
        agg_path = out_dir / "aggregate_all_folds.csv"
        agg_df.to_csv(agg_path, index=False)
        print(f"\nСводный агрегат по датасету: {agg_path}")

    # --- Сводный файл по всем датасетам для этой модели ---
    all_dataset_aggs = []
    for dataset_dir in dataset_dirs:
        agg_path = MODEL_OUTPUT_ROOT / dataset_dir.name / "aggregate_all_folds.csv"
        if agg_path.exists():
            df = pd.read_csv(agg_path)
            df.insert(0, "dataset", dataset_dir.name)
            all_dataset_aggs.append(df)

    if all_dataset_aggs:
        combined = pd.concat(all_dataset_aggs, ignore_index=True)
        meta_cols = ["dataset", "fold"]
        other = [c for c in combined.columns if c not in meta_cols]
        combined = combined[meta_cols + other]
        combined_path = MODEL_OUTPUT_ROOT / "aggregate_all_datasets.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\nСводка по всем датасетам: {combined_path}")

    print("\nГотово.")


if __name__ == "__main__":
    main()
