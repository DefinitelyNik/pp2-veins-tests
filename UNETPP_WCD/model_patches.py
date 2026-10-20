# UNETPP_WCD/model_patches.py
import copy
import time
import pathlib
from typing import Dict, List, Optional

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
from common.topology_loss import SoftEulerCharacteristicLoss
from common.topology_metrics import TopologyErrorAccumulator

NUM_CLASSES = settings.NUM_CLASSES

# ---------------------------------------------------------------------------
# Бюджет обучения (НЕ ИЗМЕНЁН)
# ---------------------------------------------------------------------------
EPOCHS_COUNT = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Гиперпараметры топологического лосса (v2: исправлены после анализа коллапса)
# ---------------------------------------------------------------------------
TOPO_WEIGHT = 0.05              # v2: было 0.5 → очень консервативно
TOPO_WARMUP_EPOCHS = 20        # v2: было 10 → больше времени на стабилизацию
TOPO_RAMP_EPOCHS = 10          # v2: было 5 → более плавный рост
TOPO_VALUE_MODE = "soft"       # v2: "soft" (согласованное значение+градиент)
TOPO_GRADIENT_MODE = "soft"    # v2: "soft" (для value_mode="soft")
TOPO_TEMPERATURE = 0.5         # v2: было 1.0 → острее softmax, меньше смещение
TOPO_WEIGHT_TEMPERATURE = 0.3  # v2: было 0.5 → умеренное усиление редких классов
TOPO_ABSENT_CLASS_SCALE = 0.0  # v2: было 0.5 → не штрафовать отсутствующие классы
TOPO_UNION_WEIGHT = 1.0
TOPO_REDUCTION = "smooth_l1"
TOPO_REL_CLAMP = 1000.0        # v2: было 5.0 → не обрезать градиент
TOPO_TV_WEIGHT = 0.0

# v2: защита от коллапса
ADAPTIVE_TOPO_WEIGHT = True    # автоматически уменьшать вес если Dice падает
ADAPTIVE_CHECK_EPOCH = 25      # через сколько эпох проверять (после ramp)
DICE_DROP_THRESHOLD = 0.05     # на сколько может упасть Dice до адаптации

# ---------------------------------------------------------------------------
# Валидация и отбор чекпоинта
# ---------------------------------------------------------------------------
SELECT_BEST_CHECKPOINT = True
TOPO_AWARE_SELECTION = True
SELECTION_DICE_TOL = 0.01
VAL_EVERY_EPOCHS = 1
TOPO_METRICS_IN_VAL = True


train_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]

val_extra_transforms = [
    A.Normalize(mean=MEAN, std=STD),
]


# ===========================================================================
# Losses
# ===========================================================================
class WeightedCEDiceLoss(nn.Module):
    def __init__(self, class_weights=None, ce_coef=1.0, dice_coef=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_coef = ce_coef
        self.dice_coef = dice_coef

    def forward(self, logits, targets):
        return self.ce_coef * self.ce(logits, targets) + self.dice_coef * self.dice(logits, targets)


class TopoAwareCriterion(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        ce_coef: float = 1.0,
        dice_coef: float = 1.0,
        topo_weight: float = TOPO_WEIGHT,
        warmup_epochs: int = TOPO_WARMUP_EPOCHS,
        ramp_epochs: int = TOPO_RAMP_EPOCHS,
        value_mode: str = TOPO_VALUE_MODE,
        gradient_mode: str = TOPO_GRADIENT_MODE,
        temperature: float = TOPO_TEMPERATURE,
        weight_temperature: float = TOPO_WEIGHT_TEMPERATURE,
        absent_class_scale: float = TOPO_ABSENT_CLASS_SCALE,
        union_weight: float = TOPO_UNION_WEIGHT,
        reduction: str = TOPO_REDUCTION,
        rel_clamp: float = TOPO_REL_CLAMP,
        tv_weight: float = TOPO_TV_WEIGHT,
    ):
        super().__init__()
        self.seg = WeightedCEDiceLoss(
            class_weights=class_weights, ce_coef=ce_coef, dice_coef=dice_coef
        )
        self.topo = SoftEulerCharacteristicLoss(
            num_classes=NUM_CLASSES,
            class_weights=class_weights,
            weight=topo_weight,
            warmup_epochs=warmup_epochs,
            ramp_epochs=ramp_epochs,
            value_mode=value_mode,
            gradient_mode=gradient_mode,
            temperature=temperature,
            weight_temperature=weight_temperature,
            absent_class_scale=absent_class_scale,
            rel_clamp=rel_clamp,
            reduction=reduction,
            include_union=True,
            union_weight=union_weight,
            tv_weight=tv_weight,
            background_index=0,
            exclude_background=True,  # v2: исключить фон
        )

    def weight_at_epoch(self, epoch: int) -> float:
        return self.topo.weight_at_epoch(epoch)

    def forward(self, logits, targets, epoch: Optional[int] = None):
        seg_loss = self.seg(logits, targets)
        topo_loss, topo_info = self.topo(logits, targets, return_info=True)
        w = self.topo.weight if epoch is None else self.weight_at_epoch(epoch)
        total = seg_loss + w * topo_loss
        info = {
            "loss_total": float(total.detach()),
            "loss_seg": float(seg_loss.detach()),
            "loss_topo_raw": float(topo_loss.detach()),
            "topo_weight": float(w),
        }
        info.update(topo_info)
        return total, info


# ===========================================================================
# Вспомогательные функции валидации
# ===========================================================================
def _class_stats_update(preds: np.ndarray, targets: np.ndarray,
                        num_classes: int, stats: np.ndarray) -> None:
    for c in range(num_classes):
        p = (preds == c)
        t = (targets == c)
        stats[c, 0] += np.logical_and(p, t).sum()
        stats[c, 1] += np.logical_and(p, ~t).sum()
        stats[c, 2] += np.logical_and(~p, t).sum()


def _macro_dice_iou(stats: np.ndarray):
    tp, fp, fn = stats[:, 0], stats[:, 1], stats[:, 2]
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    per_class = {c: (float(dice[c]), float(iou[c])) for c in range(stats.shape[0])}
    return float(np.mean(dice)), float(np.mean(iou)), per_class


# ===========================================================================
# Model
# ===========================================================================
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

        self.criterion = TopoAwareCriterion(class_weights=class_weights)
        print(f"[topo] {self.criterion.topo}")

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = None
        self.LEARNING_RATE = LEARNING_RATE
        self.is_fitted = False
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    def _build_class_weights(self) -> torch.Tensor:
        print(f'Building class weights')
        mask_paths = sorted(settings.TRAIN_MASK_DIR.glob("*.png"))
        class_counts = np.zeros(NUM_CLASSES, dtype=np.float64)
        color_map = {k: np.array(v, dtype=np.uint8) for k, v in settings.COLOR_MAP.items()}

        for mask_path in mask_paths:
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
        print(f'Weights are built: '
              f'{ {settings.CLASS_NAMES[i]: round(float(weights[i]), 4) for i in range(NUM_CLASSES)} }')
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

    # ------------------------------------------------------------------
    def _evaluate_loss(self, loader) -> float:
        self.model.eval()
        loss_total = 0.0
        with torch.no_grad():
            for images, masks in loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)
                logits = self.model(images)
                loss, _ = self.criterion(logits, masks)
                loss_total += loss.item()
        self.model.train()
        return loss_total / max(len(loader), 1)

    def _cpu_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @torch.no_grad()
    def _validate(self, val_loader) -> Dict[str, float]:
        was_training = self.model.training
        self.model.eval()

        stats = np.zeros((NUM_CLASSES, 3), dtype=np.float64)
        acc = TopologyErrorAccumulator(
            NUM_CLASSES, settings.CLASS_NAMES,
            include_background=True, include_union=True,
        ) if TOPO_METRICS_IN_VAL else None

        for images, masks in val_loader:
            images = images.to(settings.DEVICE, non_blocking=True)
            logits = self.model(images)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            tgts = masks.numpy().astype(np.uint8)

            _class_stats_update(preds, tgts, NUM_CLASSES, stats)
            if acc is not None:
                acc.update(preds, tgts)

        macro_dice, macro_iou, per_class = _macro_dice_iou(stats)
        out = {"val_macro_dice": macro_dice, "val_macro_iou": macro_iou}
        for c, (d, i) in per_class.items():
            out[f"val_dice_{settings.CLASS_NAMES[c]}"] = d

        if acc is not None:
            topo = acc.compute()
            out.update(topo)
            out["val_topo_error"] = acc.summary()
        else:
            out["val_topo_error"] = float("inf")

        if was_training:
            self.model.train()
        return out

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader: Optional[torch.utils.data.DataLoader] = None,
            save_best: bool = True, patience=None, **kwargs):

        print(f'Start fitting')
        print(f'[topo v2] value_mode={TOPO_VALUE_MODE}, gradient_mode={TOPO_GRADIENT_MODE}, '
              f'weight={TOPO_WEIGHT}, warmup={TOPO_WARMUP_EPOCHS}, ramp={TOPO_RAMP_EPOCHS}, '
              f'temperature={TOPO_TEMPERATURE}, exclude_background=True')

        self._ensure_scheduler(len(train_loader))
        self.history = []

        best_dice = -1.0
        best_state_dice: Optional[Dict[str, torch.Tensor]] = None
        band_topo = float("inf")
        band_dice = -1.0
        band_state: Optional[Dict[str, torch.Tensor]] = None

        # v2: для adaptive weight adjustment
        pre_topo_dice: Optional[float] = None
        adaptive_applied = False

        start = time.time()
        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0
            ep_seg, ep_topo, ep_w = 0.0, 0.0, 0.0
            n_steps = 0

            for images, masks in train_loader:
                images = images.to(settings.DEVICE, non_blocking=True)
                masks = masks.to(settings.DEVICE, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss, info = self.criterion(logits, masks, epoch=epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                train_loss += float(info["loss_total"])
                ep_seg += float(info["loss_seg"])
                ep_topo += float(info["loss_topo_raw"])
                ep_w = float(info["topo_weight"])
                n_steps += 1

            n_steps = max(n_steps, 1)
            train_loss /= n_steps
            ep_seg /= n_steps
            ep_topo /= n_steps

            current_lr = self.optimizer.param_groups[0]["lr"]
            approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
            time_left_str = utils.beautify_time_left(approx_time_left)

            record = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_seg_loss": ep_seg,
                "train_topo_loss_raw": ep_topo,
                "topo_weight": ep_w,
            }

            # ---------------- валидация ----------------
            do_val = (val_loader is not None) and ((epoch + 1) % VAL_EVERY_EPOCHS == 0)
            if do_val:
                vm = self._validate(val_loader)
                record.update(vm)
                v_dice = vm["val_macro_dice"]
                v_topo = vm["val_topo_error"]

                # v2: запомнить Dice в конце warmup (перед включением topo)
                if epoch == TOPO_WARMUP_EPOCHS - 1:
                    pre_topo_dice = v_dice
                    print(f"[topo v2] Запомнил Dice перед topo: {pre_topo_dice:.4f}")

                # v2: adaptive weight adjustment
                if (ADAPTIVE_TOPO_WEIGHT and not adaptive_applied
                    and epoch == ADAPTIVE_CHECK_EPOCH - 1 and pre_topo_dice is not None):
                    dice_drop = pre_topo_dice - v_dice
                    if dice_drop > DICE_DROP_THRESHOLD:
                        old_weight = self.criterion.topo.weight
                        self.criterion.topo.weight *= 0.5
                        print(f"[topo v2] ⚠️ Dice упал на {dice_drop:.3f} "
                              f"(с {pre_topo_dice:.4f} до {v_dice:.4f}). "
                              f"Уменьшаю topo_weight: {old_weight:.3f} → {self.criterion.topo.weight:.3f}")
                        adaptive_applied = True

                if SELECT_BEST_CHECKPOINT:
                    if v_dice > best_dice:
                        best_dice = v_dice
                        best_state_dice = self._cpu_state()

                    if (TOPO_AWARE_SELECTION
                            and v_dice >= best_dice - SELECTION_DICE_TOL
                            and v_topo < band_topo):
                        band_topo = v_topo
                        band_dice = v_dice
                        band_state = self._cpu_state()

                topo_str = (f" | Val Dice: {v_dice:.4f} | Val TopoErr: {v_topo:.3f}"
                            f" | bestDice: {best_dice:.4f}")
            else:
                topo_str = ""

            self.history.append(record)
            print(f"Epoch {epoch+1}/{EPOCHS_COUNT} | "
                  f"Train Loss: {train_loss:.4f} (seg {ep_seg:.4f} + "
                  f"{ep_w:.2f}·topo {ep_topo:.4f})"
                  f"{topo_str} | Left approx {time_left_str}")

        # ---------------- финальный выбор весов ----------------
        if SELECT_BEST_CHECKPOINT and (best_state_dice is not None or band_state is not None):
            use_band = (
                TOPO_AWARE_SELECTION
                and band_state is not None
                and band_dice >= best_dice - SELECTION_DICE_TOL
            )
            if use_band:
                chosen, why = band_state, (
                    f"topology-aware (Dice {band_dice:.4f} в полосе "
                    f"best-{SELECTION_DICE_TOL}, min TopoErr {band_topo:.3f})"
                )
            else:
                chosen, why = best_state_dice, f"best val Dice {best_dice:.4f}"
            self.model.load_state_dict(chosen)
            print(f"[checkpoint] восстановлены веса: {why}")

        self.is_fitted = True

    # ------------------------------------------------------------------
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE, non_blocking=True)
            logits = self.model(images)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            images = images.to(settings.DEVICE, non_blocking=True)
            return self.model(images).cpu()

    def save(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "num_classes": NUM_CLASSES,
            "LEARNING_RATE": self.LEARNING_RATE,
            "encoder_name": ENCODER_NAME,
            "class_weights": self.class_weights.detach().cpu(),
            "history": self.history,
            "topo_config": {
                "weight": TOPO_WEIGHT,
                "warmup_epochs": TOPO_WARMUP_EPOCHS,
                "ramp_epochs": TOPO_RAMP_EPOCHS,
                "value_mode": TOPO_VALUE_MODE,
                "gradient_mode": TOPO_GRADIENT_MODE,
                "weight_temperature": TOPO_WEIGHT_TEMPERATURE,
                "absent_class_scale": TOPO_ABSENT_CLASS_SCALE,
                "union_weight": TOPO_UNION_WEIGHT,
                "reduction": TOPO_REDUCTION,
                "exclude_background": True,
            },
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=settings.DEVICE)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception:
                pass
        self.history = checkpoint.get("history", [])
        self.is_fitted = True

    # ------------------------------------------------------------------
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
    def get_training_history_save_path() -> pathlib.Path:
        return pathlib.Path("UNETPP_WCD/train_history_patches.csv")

    @staticmethod
    def get_model_name():
        return 'UNET-WCD-TOPO-v2'

    @staticmethod
    def get_val_viz_path() -> pathlib.Path:
        return pathlib.Path('UNETPP_WCD/val_viz_patches')