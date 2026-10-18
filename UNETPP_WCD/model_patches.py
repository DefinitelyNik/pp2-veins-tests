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

from UNETPP_WCD.topological_loss import WeightedFocalDiceTopoLoss

NUM_CLASSES = settings.NUM_CLASSES
EPOCHS_COUNT = 50                # ← фиксировано 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ─── Гиперпараметры лосса ────────────────────────────────────────────
FOCAL_GAMMA = 2.0
FOCAL_COEF = 1.0
DICE_COEF = 2.0                  # усилен: редкие классы важнее
TOPO_COEF = 0.001                # очень мягкий, чтобы не ломать сегментацию
TOPO_TEMPERATURE = 1.0
TOPO_WARMUP_EPOCHS = 5           # 5 эпох без Topo
TOPO_RAMPUP_EPOCHS = 7           # эпохи 6–12: плавный рост
# ──────────────────────────────────────────────────────────────────────

train_extra_transforms = [A.Normalize(mean=MEAN, std=STD)]
val_extra_transforms = [A.Normalize(mean=MEAN, std=STD)]


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

    # ── Агрессивные веса классов ──────────────────────────────────────
    def _build_class_weights(self) -> torch.Tensor:
        print('Building class weights')
        mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        color_map = {k: np.array(v, dtype=np.uint8)
                     for k, v in settings.COLOR_MAP.items()}

        for mask_path in mask_paths:
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask_labels = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
            for cls_idx, color in color_map.items():
                mask_labels[np.all(mask_rgb == color, axis=-1)] = cls_idx
            class_counts += np.bincount(mask_labels.reshape(-1),
                                        minlength=NUM_CLASSES)

        # Effective-number weights (beta ближе к 1 → сильнее давим редкие)
        beta = 0.99999
        eff = 1.0 - np.power(beta, class_counts)
        w_eff = (1.0 - beta) / np.maximum(eff, 1e-12)
        w_eff /= w_eff.mean()

        # Frequency weights
        median_freq = np.median(class_counts)
        freq = class_counts / class_counts.sum()
        w_freq = median_freq / (freq * NUM_CLASSES + 1e-10)

        # Комбинация
        w = 0.5 * w_eff + 0.5 * w_freq
        w /= w.sum()
        w *= NUM_CLASSES

        print(f'  counts : {class_counts}')
        print(f'  weights: {np.round(w, 4)}')
        return torch.tensor(w, dtype=torch.float32)
    # ────────────────────────────────────────────────────────────────────

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

    def fit(self, train_loader, val_loader=None,
            save_best=True, patience=None, **kwargs):
        print(f'Start fitting  ({EPOCHS_COUNT} epochs)')
        self._ensure_scheduler(len(train_loader))
        start = time.time()

        for epoch in range(EPOCHS_COUNT):
            self.criterion.set_epoch(epoch)
            self.model.train()

            ep_focal = ep_dice = ep_topo = ep_total = 0.0
            n = 0

            for images, masks in train_loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                comp = self.criterion.get_loss_components(logits, masks)
                ep_focal += comp['focal']
                ep_dice += comp['dice']
                ep_topo += comp['topo']
                ep_total += loss.item()
                n += 1

            ep_focal /= max(n, 1)
            ep_dice /= max(n, 1)
            ep_topo /= max(n, 1)
            ep_total /= max(n, 1)

            w = self.criterion._topo_weight()
            if epoch < TOPO_WARMUP_EPOCHS:
                tag = "warmup"
            elif w < TOPO_COEF:
                tag = f"rampup {w:.5f}"
            else:
                tag = f"ACTIVE  {w:.5f}"

            elapsed = time.time() - start
            left = elapsed / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)

            print(
                f"Epoch {epoch+1:2d}/{EPOCHS_COUNT} | "
                f"Loss: {ep_total:.4f}  "
                f"(Focal: {ep_focal:.4f}  Dice: {ep_dice:.4f}  "
                f"Topo[{tag}]: {ep_topo:.2f}) | "
                f"Left: {utils.beautify_time_left(left)}"
            )

        self.is_fitted = True

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE, non_blocking=True)
            return torch.argmax(self.model(images), dim=1).cpu()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": NUM_CLASSES,
            "LEARNING_RATE": self.LEARNING_RATE,
            "encoder_name": ENCODER_NAME,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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
        return 'UNET-WCD-FocalTopo'

    @staticmethod
    def get_val_viz_path() -> pathlib.Path:
        return pathlib.Path('UNETPP_WCD/val_viz_patches')