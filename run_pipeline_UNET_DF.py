import pandas as pd
import torch
import common.settings as settings
import common.data_preparation as data_prep
from common import metrics

# --- ЗДЕСЬ ВЫБИРАЕМ МОДЕЛЬ ---
# Импортируем модуль модели и подставляем его в переменную ModelClass
from UNETPP_WCT.model import UNetPPWCT, train_extra_transforms, val_extra_transforms
ModelClass = UNetPPWCT

# --- ПАРАМЕТРЫ ПАЙПЛАЙНА ---
USE_FITTED = False               # False – обучить, True – загрузить готовую
INTERVAL_METRICS_SAVE_PATH = ModelClass.get_interval_metrics_save_path()
RAW_METRICS_SAVE_PATH = ModelClass.get_raw_metrics_save_path()

VISUALIZE = True  # Показывать ли для визуального сравнения реальные данные и что предсказала модель

# --- ЗАГРУЗКА ДАННЫХ ---
print("Загрузка данных...")
loaders = data_prep.create_cross_val_loaders(
    train_extra_transforms=train_extra_transforms,
    val_extra_transforms=val_extra_transforms,
)

# --- МОДЕЛЬ ---
models = [ModelClass() for _ in range(settings.FOLDS_NUM)]
if not USE_FITTED:
    print("Обучение модели...")
    for i, (model, (train_loader, val_loader)) in enumerate(zip(models, loaders), start=1):
        print(f'Бакет {i}...')
        model.fit(train_loader, save_best=True)

        model_save_path = ModelClass.get_model_save_path(i)
        model.save(model_save_path)
        print(f"Модель сохранена в {model_save_path}")
else:
    for i, model in enumerate(models, start=1):
        model_save_path = ModelClass.get_model_save_path(i)

        print(f"Загрузка предобученной модели из {model_save_path}")
        model.load(model_save_path)

# --- ПРЕДСКАЗАНИЯ НА ТЕСТОВЫХ ДАННЫХ ---
print("Выполнение предсказаний...")
raw_metrics = []
for i, (model, test_loader) in enumerate(zip(models, [loader[1] for loader in loaders]), start=1):
    print(f'Для модели {i} ...')
    all_preds = []
    all_targets = []
    for images, masks in test_loader:
        preds = model.predict(images)  # (B, H, W) long
        all_preds.append(preds.view(-1))
        all_targets.append(masks.view(-1))

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    raw_metrics.append(metrics.get_all_metrics(all_preds, all_targets, num_classes=5))

raw_df = pd.DataFrame(raw_metrics)
raw_df.to_csv(RAW_METRICS_SAVE_PATH, index=False)

interval_df = metrics.get_interval_metrics_from_raw(raw_df)
interval_df.to_csv(INTERVAL_METRICS_SAVE_PATH, index=False)
print(f"Сырые метрики сохранены в {RAW_METRICS_SAVE_PATH}")
print(f"Интервальные оценки сохранены в {INTERVAL_METRICS_SAVE_PATH}")
print("Готово.")
