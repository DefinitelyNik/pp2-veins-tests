import pathlib
from typing import Optional

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim

from common import settings
from common.base_model import BaseModel
from common.topological_loss import TopologicalVesselLoss


NUM_CLASSES = settings.NUM_CLASSES
IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class CombinedLoss(nn.Module):
    """Focal + Dice: без CrossEntropyLoss, с упором на редкие и трудные классы."""

    def __init__(self, focal_weight: float = 0.6, dice_weight: float = 0.3, topo_weight: float = 0.1):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode="multiclass", gamma=2.0)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

        # Инициализация топологического лосса
        self.topo = TopologicalVesselLoss(num_classes=settings.NUM_CLASSES)

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.topo_weight = topo_weight

    def forward(self, logits, targets):
        loss_focal = self.focal(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_topo = self.topo(logits, targets)

        return (self.focal_weight * loss_focal +
                self.dice_weight * loss_dice +
                self.topo_weight * loss_topo)


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
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = None
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False

    def _build_class_weights(self) -> torch.Tensor:
        """
        Считаем частоты классов в train для диагностики дисбаланса.
        Подготовку данных и pipeline не меняем.
        """
        mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)

        color_map = {k: np.array(v, dtype=np.uint8) for k, v in settings.COLOR_MAP.items()}

        for mask_path in mask_paths:
            from PIL import Image
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask_labels = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for class_idx, color in color_map.items():
                mask_labels[np.all(mask_rgb == color, axis=-1)] = class_idx
            bincount = np.bincount(mask_labels.reshape(-1), minlength=NUM_CLASSES)
            class_counts += bincount

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-12)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

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

        if val_loader is None:
            val_loader = train_loader

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._ensure_scheduler(len(train_loader))

        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0

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

            train_loss /= max(len(train_loader), 1)
            val_loss = self._evaluate_loss(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch + 1}/{EPOCHS_COUNT} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}"
            )

            if save_best and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if save_best and best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(settings.DEVICE)

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
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.is_fitted = True

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return pathlib.Path(f"IMPROVED_ARTEM_UNETPP/weights_{fold_num} copy.pth")

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("IMPROVED_ARTEM_UNETPP/interval_metrics.csv")

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("IMPROVED_ARTEM_UNETPP/raw_metrics.csv")

    @staticmethod
    def get_model_name():
        return 'UN-DF'

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        img = self._denormalize(image_tensor)
        true_rgb = self._mask_to_rgb(mask_tensor)
        with torch.no_grad():
            pred = self.predict(image_tensor.unsqueeze(0)).squeeze(0).cpu()
        pred_rgb = self._mask_to_rgb(pred)

        ax_image.imshow(img)
        ax_image.axis("off")
        ax_truth.imshow(true_rgb)
        ax_truth.axis("off")
        ax_pred.imshow(pred_rgb)
        ax_pred.axis("off")

    def _denormalize(self, img_tensor):
        img = img_tensor.detach().cpu().numpy().transpose(1, 2, 0)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = img * std + mean
        return np.clip(img, 0, 1)

    def _mask_to_rgb(self, mask_tensor):
        mask = mask_tensor.detach().cpu().numpy().astype(np.uint8)
        rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb