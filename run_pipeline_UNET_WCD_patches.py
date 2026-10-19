# run_pipeline_UNET_WCD_patches.py
from pathlib import Path

import pandas as pd

import common.settings_patches as settings
import common.data_preparation_patches as data_prep
from common import metrics_patches as mp

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
from UNETPP_WCD.model_patches import (
    ImprovedUNetPlusPlus, train_extra_transforms, val_extra_transforms
)
ModelClass = ImprovedUNetPlusPlus

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = False
INTERVAL_METRICS_SAVE_PATH = ModelClass.get_interval_metrics_save_path()
RAW_METRICS_SAVE_PATH = ModelClass.get_raw_metrics_save_path()
TRAIN_HISTORY_SAVE_PATH = ModelClass.get_training_history_save_path()

VISUALIZE = True
VIS_DIR = ModelClass.get_val_viz_path()

# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
loaders = data_prep.create_cross_val_loaders(
    train_extra_transforms=train_extra_transforms,
    val_extra_transforms=val_extra_transforms,
)

val_source_keys_per_fold = data_prep.get_val_source_keys_per_fold()
assert len(val_source_keys_per_fold) == settings.FOLDS_NUM

# --- МОДЕЛЬ ---
models = [ModelClass() for _ in range(settings.FOLDS_NUM)]

if not USE_FITTED:
    print("Обучение модели...")
    for i, (model, (train_loader, val_loader)) in enumerate(
        zip(models, loaders), start=1
    ):
        print(f"Фолд {i}...")
        # ВАЖНО: val_loader теперь передаётся — без него не работает
        # topology-aware отбор чекпоинта (min Δβ within top-K Dice).
        model.fit(train_loader, val_loader, save_best=True)
        save_path = ModelClass.get_model_save_path(i)
        model.save(save_path)
        print(f"Модель сохранена в {save_path}")
else:
    for i, model in enumerate(models, start=1):
        save_path = ModelClass.get_model_save_path(i)
        print(f"Загрузка предобученной модели из {save_path}")
        model.load(save_path)

# --- ИСТОРИЯ ОБУЧЕНИЯ (loss-компоненты + val Dice/Δβ по эпохам) ---
TRAIN_HISTORY_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
hist_frames = []
for i, model in enumerate(models, start=1):
    h = getattr(model, "history", None)
    if h:
        df = pd.DataFrame(h)
        df.insert(0, "fold", i)
        hist_frames.append(df)
if hist_frames:
    pd.concat(hist_frames, ignore_index=True).to_csv(TRAIN_HISTORY_SAVE_PATH, index=False)
    print(f"История обучения сохранена в {TRAIN_HISTORY_SAVE_PATH}")

# --- МЕТРИКИ ПО ПАТЧАМ ---
print("\n=== Метрики по патчам ===")
raw_metrics_patches = []
for fold_idx, (model, (train_loader, val_loader)) in enumerate(
    zip(models, loaders), start=1
):
    print(f"Фолд {fold_idx} (patches):")

    metrics_dict = mp.evaluate_fold_by_patches(
        model=model,
        val_loader=val_loader,
        num_classes=settings.NUM_CLASSES,
        class_names=settings.CLASS_NAMES,
        verbose=True,
    )
    raw_metrics_patches.append(metrics_dict)

raw_patches_df = pd.DataFrame(raw_metrics_patches)
raw_patches_df.to_csv(RAW_METRICS_SAVE_PATH, index=False)

interval_patches_df = mp.get_interval_metrics_from_raw(raw_patches_df)
interval_patches_df.to_csv(INTERVAL_METRICS_SAVE_PATH, index=False)
print(f"Метрики по патчам сохранены в {RAW_METRICS_SAVE_PATH}")

# --- МЕТРИКИ ПО ВОССТАНОВЛЕННЫМ ИЗОБРАЖЕНИЯМ ---
print("\n=== Метрики по восстановленным изображениям (без перекрытия) ===")
raw_metrics_recon = []
for fold_idx, (model, val_keys) in enumerate(
    zip(models, val_source_keys_per_fold), start=1
):
    print(f"Фолд {fold_idx} (reconstructed): val-источников = {len(val_keys)}")

    metrics_dict, visualizations = mp.evaluate_fold_reconstructed(
        model=model,
        val_source_keys=val_keys,
        normalized_img_dir=settings.NORMALIZED_IMG_DIR,
        normalized_msk_dir=settings.NORMALIZED_MSK_DIR,
        num_classes=settings.NUM_CLASSES,
        color_map=settings.COLOR_MAP,
        class_names=settings.CLASS_NAMES,
        val_extra_transforms=val_extra_transforms,
        patch_size=settings.PATCH_SIZE,
        stride=settings.STRIDE,
        batch_size=settings.BATCH_SIZE,
        verbose=True,
    )
    raw_metrics_recon.append(metrics_dict)

    if VISUALIZE:
        for src, (pred, gt) in visualizations.items():
            out_path = VIS_DIR / f"fold{fold_idx}_{src}.png"
            mp.save_visualization(pred, gt, out_path, settings.COLOR_MAP)
        print(f"  визуализации сохранены в {VIS_DIR}")

raw_recon_df = pd.DataFrame(raw_metrics_recon)
recon_csv = RAW_METRICS_SAVE_PATH.with_name(
    RAW_METRICS_SAVE_PATH.stem + "_reconstructed.csv"
)
raw_recon_df.to_csv(recon_csv, index=False)

interval_recon_df = mp.get_interval_metrics_from_raw(raw_recon_df)
interval_recon_csv = INTERVAL_METRICS_SAVE_PATH.with_name(
    INTERVAL_METRICS_SAVE_PATH.stem + "_reconstructed.csv"
)
interval_recon_df.to_csv(interval_recon_csv, index=False)
print(f"Метрики по восстановленным сохранены в {recon_csv}")

print("\nГотово.")