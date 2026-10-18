import time
import pathlib
from typing import Optional
import albumentations as A
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from common import settings_patches as settings
from common.base_model import BaseModel
from common import utils

# НОВЫЙ ИМПОРТ
from UNETPP_WCD.topological_loss import WeightedFocalDiceTopoLoss

NUM_CLASSES = settings.NUM_CLASSES
EPOCHS_COUNT = 100  # <-- УВЕЛИЧЕНО с 50 до 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ГИПЕРПАРАМЕТРЫ для улучшенного лосса
FOCAL_GAMMA = 2.0  # Focal loss focusing parameter
FOCAL_COEF = 1.0  # Вес Focal Loss
DICE_COEF = 1.5  # <-- УВЕЛИЧЕН с 1.0 до 1.5 (Dice важнее)
TOPO_COEF = 0.005  # <-- УМЕНЬШЕН с 0.1 до 0.005
TOPO_TEMPERATURE = 1.0
TOPO_WARMUP_EPOCHS = 15  # <-- УВЕЛИЧЕН с 10 до 15
TOPO_RAMPUP_EPOCHS = 15  # <-- УВЕЛИЧЕН с 10 до 15

train_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]


class ImprovedUNetPlusPlus(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=NUM_CLASSES,
        ).to(settings.DEVICE)

        class_weights = self._build_class_weights().to(settings.DEVICE)
        self.class_weights = class_weights

        # ИСПОЛЬЗУЕМ УЛУЧШЕННЫЙ ЛОСС
        self.criterion = WeightedFocalDiceTopoLoss(
            class_weights=class_weights,
            focal_gamma=FOCAL_GAMMA,
            focal_coef=FOCAL_COEF,
            dice_coef=DICE_COEF,
            topo_coef=TOPO_COEF,
            topo_temperature=TOPO_TEMPERATURE,
            topo_warmup_epochs=TOPO_WARMUP_EPOCHS,
            topo_rampup_epochs=TOPO_RAMPUP_EPOCHS,
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = None
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False

    def _build_class_weights(self) -> torch.Tensor:
        print(f'Building class weights')
        mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        color_map = {k: np.array(v, dtype=np.uint8) for k, v in settings.COLOR_MAP.items()}
        for i, mask_path in enumerate(mask_paths, start=1):
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask_labels = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for class_idx, color in color_map.items():
                mask_labels[np.all(mask_rgb == color, axis=-1)] = class_idx
            bincount = np.bincount(mask_labels.reshape(-1), minlength=NUM_CLASSES)
            class_counts += bincount

        # БОЛЕЕ АГРЕССИВНЫЕ ВЕСА
        beta = 0.99999  # <-- УВЕЛИЧЕН с 0.9999 до 0.99999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / weights.mean()

        # Дополнительное усиление для редких классов
        median_freq = np.median(class_counts)
        class_freq = class_counts / class_counts.sum()
        freq_weights = median_freq / (class_freq * NUM_CLASSES + 1e-10)

        # Комбинируем effective number weights и frequency weights
        combined_weights = (weights * 0.5 + freq_weights * 0.5)
        combined_weights = combined_weights / combined_weights.sum() * NUM_CLASSES

        print(f'Class counts: {class_counts}')
        print(f'Class weights: {combined_weights}')
        return torch.tensor(combined_weights, dtype=torch.float32)

    def _ensure_scheduler(self, train_loader_len: int) -> None:
        if self.scheduler is None:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.LEARNING_RATE,
                epochs=EPOCHS_COUNT,
                steps_per_epoch=train_loader_len,
                pct_start=0.15,
                anneal_strategy="cos",
                div_factor=10.0,
                final_div_factor=100.0,
            )

    def _evaluate_loss(self, loader) -> float:
        self.model.eval()
        loss_total = 0.0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)
                logits = self.model(images)
                loss_total += self.criterion(logits, masks).item()
        return loss_total / max(len(loader), 1)

    def fit(self, train_loader, val_loader: Optional[torch.utils.data.DataLoader] = None,
            save_best: bool = True, patience=None, **kwargs):
        print(f'Start fitting')
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._ensure_scheduler(len(train_loader))
        start = time.time()

        for epoch in range(EPOCHS_COUNT):
            self.criterion.set_epoch(epoch)

            self.model.train()
            train_loss = 0.0

            epoch_focal = 0.0
            epoch_dice = 0.0
            epoch_topo = 0.0
            num_batches = 0

            for images, masks in train_loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()

                components = self.criterion.get_loss_components(logits, masks)
                epoch_focal += components['focal']
                epoch_dice += components['dice']
                epoch_topo += components['topo']
                num_batches += 1

            train_loss /= max(len(train_loader), 1)
            epoch_focal /= max(num_batches, 1)
            epoch_dice /= max(num_batches, 1)
            epoch_topo /= max(num_batches, 1)

            current_lr = self.optimizer.param_groups[0]["lr"]
            approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
            time_left_str = utils.beautify_time_left(approx_time_left)

            topo_status = f"rampup({components['topo_coef']:.4f})" if epoch >= TOPO_WARMUP_EPOCHS and components[
                'topo_coef'] < TOPO_COEF else "ACTIVE"
            if epoch < TOPO_WARMUP_EPOCHS:
                topo_status = "warmup"

            print(
                f"Epoch {epoch + 1}/{EPOCHS_COUNT} | "
                f"Loss: {train_loss:.4f} (Focal: {epoch_focal:.4f}, Dice: {epoch_dice:.4f}, Topo[{topo_status}]: {epoch_topo:.4f}) | "
                f"Left: {time_left_str}"
            )

        self.is_fitted = True

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE, non_blocking=True)
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": NUM_CLASSES,
            "LEARNING_RATE": self.LEARNING_RATE,
            "encoder_name": ENCODER_NAME,
            "focal_gamma": FOCAL_GAMMA,
            "dice_coef": DICE_COEF,
            "topo_coef": TOPO_COEF,
            "topo_temperature": TOPO_TEMPERATURE,
            "topo_warmup_epochs": TOPO_WARMUP_EPOCHS,
            "topo_rampup_epochs": TOPO_RAMPUP_EPOCHS,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.is_fitted = True

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return pathlib.Path(f"UNETPP_WCD/weights_{fold_num}_patches.pth")

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_WCD/interval_metrics_patches.csv")

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_WCD/raw_metrics_patches.csv")

    @staticmethod
    def get_model_name():
        return 'UNET-WCD-Focal-Topo'

    @staticmethod
    def get_val_viz_path() -> pathlib.Path:
        return pathlib.Path('UNETPP_WCD/val_viz_patches')