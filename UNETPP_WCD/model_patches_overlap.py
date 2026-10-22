"""UNet++: train-only CE weights, foreground/union Dice and soft clDice.

Place in UNETPP_WCD/model_patches_overlap.py. No dependency on the unprovided
BaseModel implementation. Public fit/predict/save/load/path methods are retained.
Checkpoints are inference checkpoints; exact training resume is NOT supported.
"""
import pathlib
import time
import warnings

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import settings_patches_overlap as settings
from common.experiment_utils_overlap import read_rgb, strict_rgb_to_labels

NUM_CLASSES = settings.NUM_CLASSES
EPOCHS_COUNT = settings.EPOCHS_COUNT
LEARNING_RATE = settings.LEARNING_RATE
WEIGHT_DECAY = settings.WEIGHT_DECAY
ENCODER_NAME = settings.ENCODER_NAME
ENCODER_WEIGHTS = settings.ENCODER_WEIGHTS
MEAN, STD = settings.MEAN, settings.STD

train_extra_transforms = []
if settings.USE_PHOTOMETRIC_AUG:
    train_extra_transforms.extend([
        A.RandomBrightnessContrast(
            brightness_limit=0.10, contrast_limit=0.10, p=0.30
        ),
        A.HueSaturationValue(
            hue_shift_limit=0, sat_shift_limit=5, val_shift_limit=5, p=0.20
        ),
    ])
train_extra_transforms.append(A.Normalize(mean=MEAN, std=STD))
val_extra_transforms = [A.Normalize(mean=MEAN, std=STD)]


def soft_erode(x):
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def soft_open(x):
    return F.max_pool2d(soft_erode(x), 3, stride=1, padding=1)


def soft_skeletonize(x, iterations):
    skeleton = F.relu(x - soft_open(x))
    for _ in range(iterations):
        x = soft_erode(x)
        delta = F.relu(x - soft_open(x))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(prob, target, iterations=10, eps=1e-6):
    pred_skeleton = soft_skeletonize(prob, iterations)
    with torch.no_grad():
        target_skeleton = soft_skeletonize(target, iterations)
    dims = (1, 2, 3)
    precision = ((pred_skeleton * target).sum(dims) + eps) / (
        pred_skeleton.sum(dims) + eps
    )
    recall = ((target_skeleton * prob).sum(dims) + eps) / (
        target_skeleton.sum(dims) + eps
    )
    score = 2 * precision * recall / (precision + recall + eps)
    present = target.sum(dims) > 0
    return 1 - score[present].mean() if present.any() else prob.sum() * 0


class VesselAwareLoss(nn.Module):
    def __init__(self, class_weights, union_coef=0.25, topology_coef=0.10,
                 skeleton_iterations=10):
        super().__init__()
        self.register_buffer("class_weights", class_weights.clone().float())
        self.union_coef = float(union_coef)
        self.topology_coef = float(topology_coef)
        self.skeleton_iterations = int(skeleton_iterations)
        self.topology_scale = 0.0

    def forward(self, logits, targets):
        logits, targets = logits.float(), targets.long()
        ce = F.cross_entropy(logits, targets, weight=self.class_weights)
        probs = logits.softmax(1)
        target_one_hot = F.one_hot(
            targets, num_classes=logits.shape[1]
        ).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        pred_fg, target_fg = probs[:, 1:], target_one_hot[:, 1:]
        target_sum = target_fg.sum(dims)
        intersection = (pred_fg * target_fg).sum(dims)
        dice = (2 * intersection + 1e-6) / (
            pred_fg.sum(dims) + target_sum + 1e-6
        )
        present = target_sum > 0
        fg_loss = 1 - dice[present].mean() if present.any() else probs.sum() * 0
        loss = ce + fg_loss
        vessel_prob = probs[:, 1:].sum(1, keepdim=True)
        vessel_target = (targets > 0).unsqueeze(1).float()
        if self.union_coef:
            intersection = (vessel_prob * vessel_target).sum()
            union_dice = (2 * intersection + 1e-6) / (
                vessel_prob.sum() + vessel_target.sum() + 1e-6
            )
            loss = loss + self.union_coef * (1 - union_dice)
        if self.topology_coef and self.topology_scale:
            loss = loss + self.topology_coef * self.topology_scale * soft_cldice_loss(
                vessel_prob, vessel_target, self.skeleton_iterations
            )
        return loss

class ImprovedUNetPlusPlus:
    def __init__(self, union_coef=None, topology_coef=None,
                 encoder_weights=ENCODER_WEIGHTS, device=None):
        settings.validate_settings()
        self.device = torch.device(settings.DEVICE if device is None else device)
        self.union_coef = settings.UNION_COEF if union_coef is None else union_coef
        self.topology_coef = settings.TOPOLOGY_COEF if topology_coef is None else topology_coef
        if self.union_coef < 0 or self.topology_coef < 0:
            raise ValueError("Loss coefficients must be nonnegative.")
        self.encoder_weights = encoder_weights
        self.model = smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME, encoder_weights=encoder_weights,
            in_channels=3, classes=NUM_CLASSES,
        ).to(self.device)
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.class_weights = None
        self.class_counts = None
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False
        self.trained_epochs = 0
        self.selected_epoch = None
        self.history = []
        self.metadata = {}

    def _build_class_weights(self, train_loader):
        paths = getattr(train_loader.dataset, "mask_paths", None)
        if paths is None:
            raise TypeError("Expected SegmentationDataset with train-only mask_paths.")
        counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        for path in paths:
            labels = strict_rgb_to_labels(read_rgb(path))
            counts += np.bincount(labels.ravel(), minlength=NUM_CLASSES)
        if counts.sum() == 0:
            raise ValueError("Empty training masks.")
        present = counts > 0
        if not present.all():
            warnings.warn(f"Classes absent in train: {np.flatnonzero(~present).tolist()}")
        frequencies = counts / counts.sum()
        weights = np.ones(NUM_CLASSES, dtype=np.float64)
        weights[present] = 1 / np.sqrt(frequencies[present])
        weights[present] /= weights[present].mean()
        weights = np.clip(weights, 0.25, 4.0)
        self.class_counts = counts.tolist()
        print("Train class pixels:", self.class_counts)
        print("CE weights:", weights.tolist())
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

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

    @torch.inference_mode()
    def _validation_score(self, loader):
        self.model.eval()
        cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=self.device)
        for images, masks in loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True).long()
            pred = self.model(images).argmax(1)
            index = masks.flatten() * NUM_CLASSES + pred.flatten()
            cm += torch.bincount(index, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        denominator = cm.sum(0) + cm.sum(1)
        dice = 2 * cm.diag().double() / denominator.clamp_min(1)
        return dice[1:].mean().item()

    def fit(self, train_loader, val_loader=None, save_best=False, patience=None,
            on_epoch_end=None, **kwargs):
        if kwargs:
            raise TypeError(f"Unsupported fit arguments: {sorted(kwargs)}")
        if self.is_fitted or self.trained_epochs:
            raise RuntimeError("Create a fresh model; resume is not implemented.")
        if not len(train_loader):
            raise ValueError("Empty train_loader.")
        if patience is not None:
            raise ValueError("Early stopping is disabled: exactly 50 epochs required.")
        if save_best and val_loader is None:
            raise ValueError("save_best requires INNER validation, never the outer test fold.")
        self.class_weights = self._build_class_weights(train_loader)
        self.criterion = VesselAwareLoss(
            self.class_weights, self.union_coef, self.topology_coef,
            settings.SKELETON_ITERATIONS,
        ).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self._ensure_scheduler(len(train_loader))
        start = time.time()
        best_score, best_state = -float("inf"), None
        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            self.criterion.topology_scale = min(1.0, max(0.0, (epoch - 4) / 10.0))
            total, samples = 0.0, 0
            for images, masks in train_loader:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True).long()
                self.optimizer.zero_grad(set_to_none=True)
                loss = self.criterion(self.model(images), masks)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                self.optimizer.step()
                self.scheduler.step()
                total += loss.item() * images.shape[0]
                samples += images.shape[0]
            self.trained_epochs = epoch + 1
            row = {
                "epoch": epoch + 1, "train_loss": total / samples,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "topology_scale": self.criterion.topology_scale,
            }
            if val_loader is not None:
                score = self._validation_score(val_loader)
                row["inner_patch_foreground_dice"] = score
                if save_best and score > best_score:
                    best_score, self.selected_epoch = score, epoch + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self.history.append(row)
            if on_epoch_end is not None:
                on_epoch_end(self.history)
            remaining = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
            print(f"Epoch {epoch+1}/50 | loss={row['train_loss']:.4f} | left={remaining/60:.1f} min")
        if save_best:
            self.model.load_state_dict(best_state)
        else:
            self.selected_epoch = EPOCHS_COUNT
        self.is_fitted = True
        return self

    @torch.inference_mode()
    def predict_proba(self, images, tta=False):
        self.model.eval()
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images Bx3xHxW, normalized with MEAN/STD.")
        images = images.to(self.device, non_blocking=True).float()
        variants = ((), (-1,), (-2,), (-2, -1)) if tta else ((),)
        accumulated = None
        for dims in variants:
            transformed = torch.flip(images, dims) if dims else images
            probs = self.model(transformed).float().softmax(1)
            if dims:
                probs = torch.flip(probs, dims)
            accumulated = probs if accumulated is None else accumulated + probs
        return (accumulated / len(variants)).cpu()

    def predict(self, images):
        return self.predict_proba(images).argmax(1)

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save({
            "format_version": 2,
            "model_state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "num_classes": NUM_CLASSES, "encoder_name": ENCODER_NAME,
            "encoder_weights": self.encoder_weights,
            "trained_epochs": self.trained_epochs, "selected_epoch": self.selected_epoch,
            "class_counts": self.class_counts,
            "class_weights": self.class_weights.detach().cpu() if self.class_weights is not None else None,
            "union_coef": self.union_coef, "topology_coef": self.topology_coef,
            "metadata": self.metadata,
        }, temporary)
        temporary.replace(path)

    def load(self, path, expected_protocol=None):
        # weights_only prevents arbitrary code execution from pickled checkpoints.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("Expected new-format checkpoint v2; original baseline files are not accepted.")
        if checkpoint.get("num_classes") != NUM_CLASSES:
            raise ValueError("Checkpoint class count mismatch.")
        if checkpoint.get("encoder_name") != ENCODER_NAME:
            raise ValueError("Checkpoint encoder mismatch.")
        metadata = checkpoint.get("metadata", {})
        if expected_protocol is not None and metadata.get("protocol_sha256") != expected_protocol:
            raise ValueError("Checkpoint dataset/split/configuration differs from this run.")
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.metadata = metadata
        self.trained_epochs = checkpoint.get("trained_epochs", 0)
        self.selected_epoch = checkpoint.get("selected_epoch")
        self.class_counts = checkpoint.get("class_counts")
        weights = checkpoint.get("class_weights")
        self.class_weights = weights.to(self.device) if weights is not None else None
        self.union_coef = checkpoint.get("union_coef", self.union_coef)
        self.topology_coef = checkpoint.get("topology_coef", self.topology_coef)
        self.encoder_weights = checkpoint.get("encoder_weights")
        self.is_fitted = True
        return self

    @staticmethod
    def get_model_save_path(fold_num: int) -> pathlib.Path:
        return settings.OUTPUT_DIR / f"weights_{fold_num}_patches_overlap.pth"

    @staticmethod
    def get_interval_metrics_save_path() -> pathlib.Path:
        return settings.OUTPUT_DIR / ("evaluation_tta" if settings.USE_TTA else "evaluation") / "interval_metrics_patches_overlap.csv"

    @staticmethod
    def get_raw_metrics_save_path() -> pathlib.Path:
        return settings.OUTPUT_DIR / ("evaluation_tta" if settings.USE_TTA else "evaluation") / "raw_metrics_patches_overlap.csv"

    @staticmethod
    def get_model_name():
        return "UNETPP-VESSEL-AWARE"

    @staticmethod
    def get_val_viz_path():
        return settings.OUTPUT_DIR / ("evaluation_tta" if settings.USE_TTA else "evaluation") / "visualizations"
