import time
import numpy as np
import pathlib

import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A

from common.base_model import BaseModel
from common import settings
from common import utils
from TFFM.tffm_model import TFFMSegmentationModel, ModelConfig

# Константы (могут быть изменены)
NUM_CLASSES = 5
IMG_SIZE = 512
EPOCHS_COUNT = 50
LEARNING_RATE = 1e-4

# Константы для модели
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Дополнительные трансформации для этой модели (без ToTensorV2, его добавит data_preparation)
train_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),  # 1=LINEAR, 0=NEAREST
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Resize(IMG_SIZE, IMG_SIZE, interpolation=1, mask_interpolation=0),
    A.Normalize(mean=MEAN, std=STD),
]


class TFFMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.config = ModelConfig(num_classes=settings.NUM_CLASSES)
        self.model = TFFMSegmentationModel(
            config=self.config,
            num_classes=settings.NUM_CLASSES,
        ).to(settings.DEVICE)
        print(settings.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        self.LEARNING_RATE = LEARNING_RATE

    def fit(self, train_loader, val_loader=None, save_best=True, patience=None, **kwargs):
        print(f'Start fitting')
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        start = time.time()
        for epoch in range(EPOCHS_COUNT):
            # Обучение
            self.model.train()
            train_loss = 0.0
            for images, masks in train_loader:
                images, masks = images.to(settings.DEVICE), masks.to(settings.DEVICE)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            # Валидация
            val_loss = None
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, masks in val_loader:
                        images, masks = images.to(settings.DEVICE), masks.to(settings.DEVICE)
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                        val_loss += loss.item()
                val_loss /= len(val_loader)

            # Вывод
            if val_loss is not None:
                print(f"Epoch {epoch+1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            else:
                approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
                time_left_str = utils.beautify_time_left(approx_time_left)
                print(f"Epoch {epoch+1}/{EPOCHS_COUNT} | Train Loss: {train_loss:.4f} | Left approx {time_left_str}")

            # Сохранение лучшей модели
            if save_best and val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                print(f"  -> New best model saved (val_loss={val_loss:.4f})")
            elif save_best and val_loss is not None and patience is not None:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Восстановление лучшей модели
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
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': settings.NUM_CLASSES,
            'LEARNING_RATE': self.LEARNING_RATE,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_classes = settings.NUM_CLASSES
        self.LEARNING_RATE = checkpoint.get('LEARNING_RATE', self.LEARNING_RATE)
        self.is_fitted = True

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return pathlib.Path(f"TFFM/weights_{fold_num}.eth")

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("TFFM/interval_metrics.csv")

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return pathlib.Path("TFFM/raw_metrics.csv")

    def visualize_sample(self, image_tensor, mask_tensor, ax_image, ax_truth, ax_pred):
        """Отрисовывает оригинал, истинную маску и предсказание на переданные оси."""
        # Денормализованное изображение (H,W,3) в диапазоне [0,1]
        img = self._denormalize(image_tensor)
        # Истинная маска в RGB
        true_rgb = self._mask_to_rgb(mask_tensor)
        # Предсказание
        with torch.no_grad():
            pred = self.predict(image_tensor.unsqueeze(0)).squeeze(0).cpu()
        pred_rgb = self._mask_to_rgb(pred)

        ax_image.imshow(img)
        ax_image.axis('off')
        ax_truth.imshow(true_rgb)
        ax_truth.axis('off')
        ax_pred.imshow(pred_rgb)
        ax_pred.axis('off')

    @staticmethod
    def get_model_name():
        return 'TFFM-CE'

    def _denormalize(self, img_tensor):
        """Преобразует нормализованный тензор (C,H,W) в numpy (H,W,3) в диапазоне [0,1]."""
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H,W,C)
        mean = np.array(MEAN)
        std = np.array(STD)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img

    def _mask_to_rgb(self, mask_tensor):
        """Преобразует маску (H,W) с индексами классов в RGB (H,W,3) uint8."""
        mask = mask_tensor.cpu().numpy().astype(np.uint8)
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in settings.COLOR_MAP.items():
            rgb[mask == cls] = color
        return rgb
