# run_pipeline_UNET_WCD_patches_overlap.py
from pathlib import Path

import pandas as pd

import common.settings_patches_overlap as settings
import common.data_preparation_patches_overlap as data_prep
from common import metrics_patches_overlap as mpo

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
from UNETPP_WCD.model_patches_overlap import (
    ImprovedUNetPlusPlus, train_extra_transforms, val_extra_transforms
)
ModelClass = ImprovedUNetPlusPlus

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = False
INTERVAL_METRICS_SAVE_PATH = ModelClass.get_interval_metrics_save_path()
RAW_METRICS_SAVE_PATH = ModelClass.get_raw_metrics_save_path()

VISUALIZE = True
VIS_DIR = ModelClass.get_val_viz_path()

NORMALIZED_IMG_DIR = Path("normalized_dataset/images")
NORMALIZED_MSK_DIR = Path("normalized_dataset/masks")

# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
loaders = data_prep.create_cross_val_loaders(
    train_extra_transforms=train_extra_transforms,
    val_extra_transforms=val_extra_transforms,
)

# Список val-источников по фолдам — критично, чтобы совпадал
# с тем, что видел create_cross_val_loaders (тот же SEED)
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
        model.fit(train_loader, save_best=True)

        model_save_path = ModelClass.get_model_save_path(i)
        model.save(model_save_path)
        print(f"Модель сохранена в {model_save_path}")
else:
    for i, model in enumerate(models, start=1):
        model_save_path = ModelClass.get_model_save_path(i)
        print(f"Загрузка предобученной модели из {model_save_path}")
        model.load(model_save_path)

# --- МЕТРИКИ ПО ВОССТАНОВЛЕННЫМ ИЗОБРАЖЕНИЯМ ---
print("Вычисление метрик по восстановленным изображениям...")
raw_metrics = []

for fold_idx, (model, val_keys) in enumerate(
    zip(models, val_source_keys_per_fold), start=1
):
    print(f"Фолд {fold_idx}: val-источников = {len(val_keys)}")

    metrics_dict, visualizations = mpo.evaluate_fold_overlap(
        model=model,
        val_source_keys=val_keys,
        normalized_img_dir=NORMALIZED_IMG_DIR,
        normalized_msk_dir=NORMALIZED_MSK_DIR,
        num_classes=settings.NUM_CLASSES,
        color_map=settings.COLOR_MAP,
        class_names=settings.CLASS_NAMES,
        val_extra_transforms=val_extra_transforms,
        patch_size=256,
        stride=128,
        priority_order=[3, 1, 2, 4, 0],
        batch_size=settings.BATCH_SIZE,
        verbose=True,
    )
    raw_metrics.append(metrics_dict)

    if VISUALIZE:
        for src, (pred, gt) in visualizations.items():
            out_path = VIS_DIR / f"fold{fold_idx}_{src}.png"
            mpo.save_visualization(pred, gt, out_path, settings.COLOR_MAP)
        print(f"  визуализации сохранены в {VIS_DIR}")

# --- СОХРАНЕНИЕ ---
raw_df = pd.DataFrame(raw_metrics)
raw_df.to_csv(RAW_METRICS_SAVE_PATH, index=False)

interval_df = mpo.get_interval_metrics_from_raw(raw_df)
interval_df.to_csv(INTERVAL_METRICS_SAVE_PATH, index=False)

print(f"Сырые метрики: {RAW_METRICS_SAVE_PATH}")
print(f"Интервальные оценки: {INTERVAL_METRICS_SAVE_PATH}")
print("Готово.")
