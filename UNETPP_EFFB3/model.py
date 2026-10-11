import time
import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A

from common.base_model import BaseModel
from common import settings
from common import utils
from common.topological_loss import TopologicalVesselLoss


# --- параметры эксперимента ---
IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4

# ImageNet-нормализация для pretrained encoder
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


# Дополнительные трансформации. ToTensorV2 добавляется в common/data_preparation.py
train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class WeightedCEDiceLoss(nn.Module):
    """
    Weighted CrossEntropy + DiceLoss для multiclass segmentation.
    CE помогает пиксельной классификации, Dice помогает при сильном дисбалансе классов.
    """

    def __init__(self, class_weights=None, ce_coef=0.9, dice_coef=0.9, topo_coef=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.topo = TopologicalVesselLoss(num_classes=settings.NUM_CLASSES)

        self.ce_coef = ce_coef
        self.dice_coef = dice_coef
        self.topo_coef = topo_coef

    def forward(self, logits, targets):
        loss_ce = self.ce(logits, targets)
        loss_dice = self.dice(logits, targets)
        loss_topo = self.topo(logits, targets)

        return self.ce_coef * loss_ce + self.dice_coef * loss_dice + self.topo_coef * loss_topo


class UNetPlusPlusEffB3(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            in_channels=3,
            classes=settings.NUM_CLASSES,
        ).to(settings.DEVICE)

        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Изначально без весов. Реальные веса считаем по train_loader внутри fit().
        self.criterion = WeightedCEDiceTopoLoss(class_weights=None)
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False

    def _compute_class_weights(self, train_loader):
        """
        Считает веса классов по пикселям train fold.
        Формула 1/sqrt(freq) мягче, чем 1/freq, поэтому не взрывает редкие классы слишком сильно.
        """
        counts = torch.zeros(settings.NUM_CLASSES, dtype=torch.float64)

        for _, masks in train_loader:
            flat = masks.view(-1).cpu()
            counts += torch.bincount(flat, minlength=settings.NUM_CLASSES).double()

        freq = counts / counts.sum().clamp_min(1.0)
        weights = 1.0 / torch.sqrt(freq.clamp_min(1e-8))

        # Нормируем, чтобы средний вес был около 1. Так learning rate остаётся адекватным.
        weights = weights / weights.mean()
        weights = weights.float().to(settings.DEVICE)

        print("Class pixel counts:", counts.long().tolist())
        print("Class weights:", [round(x, 4) for x in weights.detach().cpu().tolist()])
        return weights

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        class_weights = self._compute_class_weights(train_loader)
        self.criterion = WeightedCEDiceLoss(class_weights=class_weights).to(settings.DEVICE)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        start = time.time()

        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0

            for images, masks in train_loader:
                images = images.to(settings.DEVICE)
                masks = masks.to(settings.DEVICE)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(settings.DEVICE)
                        masks = masks.to(settings.DEVICE)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

            if val_loss is not None:
                print(f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
                time_left_str = utils.beautify_time_left(approx_time_left)
                print(f"Epoch {epoch + 1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Left approx {time_left_str}")

            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and val_loss is not None and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if save_best and best_state is not None:
            self.model.load_state_dict(best_state)
            self.model.to(settings.DEVICE)

        self.is_fitted = True

    def predict(self, images):
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE)
            outputs = self.model(images)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": settings.NUM_CLASSES,
            "LEARNING_RATE": self.LEARNING_RATE,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.num_classes = settings.NUM_CLASSES
        self.LEARNING_RATE = checkpoint.get("LEARNING_RATE", self.LEARNING_RATE)
        self.is_fitted = True

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return pathlib.Path(f"UNETPP_EFFB3/weights_{fold_num}.eth")

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_EFFB3/interval_metrics.csv")

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_EFFB3/raw_metrics.csv")

    @staticmethod
    def get_model_name():
        return 'UN-WCD'

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        img = self._denormalize(image_tensor)
        true_rgb = self._mask_to_rgb(mask_tensor)
        with torch.no_grad():
            pred = self.predict(image_tensor.unsqueeze(0)).squeeze(0).cpu()
        pred_rgb = self._mask_to_rgb(pred)

        # import matplotlib.pyplot as plt
        # plt.imshow(pred_rgb)
        # plt.show()
        from PIL import Image
        # Image.fromarray(img).save('Source.jpg')
        Image.fromarray(pred_rgb).save('UN-WCD_3.jpg')
        Image.fromarray(true_rgb).save('True_3.jpg')
        exit()

        ax_image.imshow(img)
        ax_image.axis("off")
        ax_truth.imshow(true_rgb)
        ax_truth.axis("off")
        ax_pred.imshow(pred_rgb)
        ax_pred.axis("off")

    def _denormalize(self, img_tensor):
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def _mask_to_rgb(self, mask_tensor):
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb
