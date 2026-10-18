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
from UNETPP_WCD.topology_loss import SoftEulerCharacteristicLoss

NUM_CLASSES = settings.NUM_CLASSES
EPOCHS_COUNT = 60  # Увеличено с 50 до 60 для лучшей сходимости с топологическим лоссом
LEARNING_RATE = 2e-4  # Немного снижен для стабильности с новым лоссом
WEIGHT_DECAY = 1e-4
ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Warmup эпохи для топологического лосса (из статьи)
TOPOLOGY_WARMUP_EPOCHS = 10
TOPOLOGY_LOSS_COEF = 0.1  # Коэффициент для топологического лосса

train_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]


class WeightedCEDiceTopologyLoss(nn.Module):
    """
    Комбинированный лосс: Weighted CrossEntropy + DiceLoss + Soft Euler Characteristic Loss
    для multiclass segmentation с учетом топологии.
    """

    def __init__(self, class_weights=None, ce_coef=1.0, dice_coef=1.0, topo_coef=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.topo = SoftEulerCharacteristicLoss(class_weights=class_weights, from_logits=True)
        self.ce_coef = ce_coef
        self.dice_coef = dice_coef
        self.topo_coef = topo_coef
        self.class_weights = class_weights

    def forward(self, logits, targets, epoch=0, warmup_epochs=TOPOLOGY_WARMUP_EPOCHS):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        topo_loss = self.topo(logits, targets, epoch=epoch, warmup_epochs=warmup_epochs)

        return (self.ce_coef * ce_loss +
                self.dice_coef * dice_loss +
                self.topo_coef * topo_loss)


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
        self.criterion = WeightedCEDiceTopologyLoss(
            class_weights=class_weights,
            ce_coef=1.0,
            dice_coef=1.0,
            topo_coef=TOPOLOGY_LOSS_COEF
        )
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = None
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False
        self.current_epoch = 0

    def _build_class_weights(self) -> torch.Tensor:
        print(f'Building class weights')
        """
        Считаем частоты классов в train для диагностики дисбаланса.
        Подготовку данных и pipeline не меняем.
        """
        mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        color_map = {k: np.array(v, dtype=np.uint8) for k, v in settings.COLOR_MAP.items()}
        for i, mask_path in enumerate(mask_paths, start=1):
            # print(f'Mask path: {i} / {len(mask_paths)}')
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
        print(f'Weights are built')
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
                loss_total += self.criterion(logits, masks, epoch=self.current_epoch).item()
        return loss_total / max(len(loader), 1)

    def fit(self, train_loader, val_loader: Optional[torch.utils.data.DataLoader] = None,
            save_best: bool = True, patience=None, **kwargs):
        print(f'Start fitting')
        # if val_loader is None:
        #     val_loader = train_loader
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self._ensure_scheduler(len(train_loader))
        start = time.time()
        for epoch in range(EPOCHS_COUNT):
            self.current_epoch = epoch
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, masks, epoch=epoch, warmup_epochs=TOPOLOGY_WARMUP_EPOCHS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)
            # val_loss = self._evaluate_loss(val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]
            approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
            time_left_str = utils.beautify_time_left(approx_time_left)
            print(
                f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | LR: {current_lr:.2e} | Left approx {time_left_str}")
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
        return pathlib.Path(f"UNETPP_WCD/weights_{fold_num}_patches.pth")

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_WCD/interval_metrics_patches.csv")

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_WCD/raw_metrics_patches.csv")

    @staticmethod
    def get_model_name():
        return 'UNET-WCD'

    @staticmethod
    def get_val_viz_path() -> pathlib.Path:
        return pathlib.Path('UNETPP_WCD/val_viz_patches')