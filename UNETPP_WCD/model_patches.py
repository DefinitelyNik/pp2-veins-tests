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
# Бюджет обучения — 50 эпох для сравнимости. НЕ меняем.
# ---------------------------------------------------------------------------
EPOCHS_COUNT = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

ENCODER_NAME = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# v5: схема весов классов.
#
# ДИАГНОСТИКА, КОТОРУЮ ЭТО ЧИНИТ:
#   Веса v4 (sqrt обратной частоты): bg 0.08, art 0.34, veins 0.31, ...
#   В CE вес применяется к ЦЕЛЕВОМУ классу пикселя, поэтому ошибка
#   "сосуд на фоне" штрафовалась весом фона 0.08 -> утечка сосудов:
#       arteries_precision 0.835 -> 0.785,  veins_precision 0.865 -> 0.826,
#       background_recall 0.9909 -> 0.9865.
#   Одновременно весь CE ослаб в ~9 раз (sum(freq*w) = 0.11 вместо 1.0).
#
# НОВАЯ СХЕМА: (1/freq) ** 0.15, кламп [0.4, 3.0], нормировка к среднему 1:
#   частоты:  bg 0.8889, art 0.0493, veins 0.0588, cross 0.0021, cap 0.0008
#   веса:     bg 0.53,   art 0.82,   veins 0.80,   cross 1.32,   cap 1.53
#   - фон возвращён до 0.53 -> утечки снова дорого штрафуются (precision);
#   - артерии/вены ~1.0 -> приоритет задачи;
#   - кроссинги/капилляры умеренно усилены (без экстремума 2.6 из v4).
# ---------------------------------------------------------------------------
CLASS_WEIGHT_POWER = 0.15
CLASS_WEIGHT_MIN = 0.4
CLASS_WEIGHT_MAX = 3.0

# ---------------------------------------------------------------------------
# Гиперпараметры топологического лосса — БЕЗ ИЗМЕНЕНИЙ (стабильно в v2/v4)
# ---------------------------------------------------------------------------
TOPO_WEIGHT = 0.05
TOPO_WARMUP_EPOCHS = 20
TOPO_RAMP_EPOCHS = 10
TOPO_VALUE_MODE = "soft"
TOPO_GRADIENT_MODE = "soft"
TOPO_TEMPERATURE = 0.5
TOPO_WEIGHT_TEMPERATURE = 0.3
TOPO_ABSENT_CLASS_SCALE = 0.0
TOPO_UNION_WEIGHT = 1.0
TOPO_REDUCTION = "smooth_l1"
TOPO_REL_CLAMP = 1000.0
TOPO_TV_WEIGHT = 0.0

# ---------------------------------------------------------------------------
# v5: штраф за путаницу артерия <-> вена
# ---------------------------------------------------------------------------
CONFUSION_WEIGHT = 0.2
CONFUSION_ARTERY_IDX = 1
CONFUSION_VEIN_IDX = 2

# Защита от коллапса
ADAPTIVE_TOPO_WEIGHT = True
ADAPTIVE_CHECK_EPOCH = 25
DICE_DROP_THRESHOLD = 0.05

# Валидация и отбор чекпоинта
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
    """
    Weighted CrossEntropy + DiceLoss для multiclass segmentation.
    CE помогает пиксельной классификации, Dice помогает при сильном дисбалансе классов.
    """
    def __init__(self, class_weights=None, ce_coef=1.0, dice_coef=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.ce_coef = ce_coef
        self.dice_coef = dice_coef

    def forward(self, logits, targets):
        return self.ce_coef * self.ce(logits, targets) + self.dice_coef * self.dice(logits, targets)


class ArteryVeinConfusionLoss(nn.Module):
    """
    v5: целевой штраф за путаницу артерия <-> вена.

    Мотивация: на визуализациях вдоль сосуда встречаются небольшие области,
    где артерия помечена как вена и наоборот, хотя пересечения там нет.
    Обычный CE почти не давит на такие места, когда модель "почти уверена"
    (например, p_artery = 0.9, p_vein = 0.08 -> -log(0.9) уже мал), а макрософтовый
    Dice эту локальную подмену не видит.

    Лосс напрямую минимизирует вероятность "соседнего" класса на пикселях
    истинного сосуда:

        L = 0.5 * ( mean_{p: GT=artery} P_vein(p)  +  mean_{p: GT=vein} P_artery(p) )

    Свойства:
      * диапазон [0, 1], 0 = путаницы нет;
      * НЕ затрагивает фон, кроссинги и капилляры;
      * дифференцируем, работает с эпохи 0 (это пиксельный член, как CE, —
        топологический warmup ему не нужен).
    """

    def __init__(
        self,
        weight: float = CONFUSION_WEIGHT,
        artery_idx: int = CONFUSION_ARTERY_IDX,
        vein_idx: int = CONFUSION_VEIN_IDX,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = float(weight)
        self.artery_idx = int(artery_idx)
        self.vein_idx = int(vein_idx)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.weight <= 0.0:
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        probs = torch.softmax(logits, dim=1)                    # (B,C,H,W)
        y_art = (targets == self.artery_idx).to(probs.dtype)    # (B,H,W)
        y_vein = (targets == self.vein_idx).to(probs.dtype)     # (B,H,W)

        n_art = y_art.sum().clamp(min=self.eps)
        n_vein = y_vein.sum().clamp(min=self.eps)

        # масса вероятности "чужого" класса на пикселях истинных сосудов
        art_to_vein = (probs[:, self.vein_idx] * y_art).sum() / n_art
        vein_to_art = (probs[:, self.artery_idx] * y_vein).sum() / n_vein

        return self.weight * 0.5 * (art_to_vein + vein_to_art)


class TopoAwareCriterion(nn.Module):
    """
    L = L_Dice+wCE  +  L_confusion(artery<->vein)  +  w_topo(epoch) * L_SoftEuler
    forward() -> (loss, info_dict)
    """

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
        confusion_weight: float = CONFUSION_WEIGHT,
    ):
        super().__init__()
        self.seg = WeightedCEDiceLoss(
            class_weights=class_weights, ce_coef=ce_coef, dice_coef=dice_coef
        )
        self.confusion = ArteryVeinConfusionLoss(weight=confusion_weight)
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
            exclude_background=True,
        )

    def weight_at_epoch(self, epoch: int) -> float:
        return self.topo.weight_at_epoch(epoch)

    def forward(self, logits, targets, epoch: Optional[int] = None):
        seg_loss = self.seg(logits, targets)
        conf_loss = self.confusion(logits, targets)
        topo_loss, topo_info = self.topo(logits, targets, return_info=True)

        w = self.topo.weight if epoch is None else self.weight_at_epoch(epoch)
        total = seg_loss + conf_loss + w * topo_loss

        info = {
            "loss_total": float(total.detach()),
            "loss_seg": float(seg_loss.detach()),
            "loss_confusion": float(conf_loss.detach()),
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
        """
        v5: tempered inverse-frequency веса.

        Формула:
            freq_c  = count_c / total_pixels
            w_c     = clip( (1 / freq_c) ** CLASS_WEIGHT_POWER,
                            CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX )
            w_c    /= mean(w)

        ИСТОРИЯ ВОПРОСА:
          * до v4 (beta=0.9999) все веса схлопывались в 1.0 — балансировки не было;
          * в v4 (степень 0.5) фон получил 0.08 -> утечка сосудов и падение
            precision артерий/вен на 4-5 п.п.;
          * степень 0.15 + кламп — баланс: фон 0.53 (надзор за утечками),
            приоритетные артерии/вены ~0.8, редкие классы усилены умеренно.
        """
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

        # --- v5: tempered inverse frequency ---
        total_pixels = max(class_counts.sum(), 1.0)
        freq = class_counts / total_pixels
        freq = np.maximum(freq, 1e-6)

        weights = np.power(1.0 / freq, CLASS_WEIGHT_POWER)
        weights = np.clip(weights, CLASS_WEIGHT_MIN, CLASS_WEIGHT_MAX)
        weights = weights / weights.mean()

        print(f'Class frequencies: '
              f'{ {settings.CLASS_NAMES[i]: round(float(freq[i]), 4) for i in range(NUM_CLASSES)} }')
        print(f'Class weights (inv-freq^{CLASS_WEIGHT_POWER}, '
              f'clamp [{CLASS_WEIGHT_MIN}, {CLASS_WEIGHT_MAX}]): '
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
        print(f'[v5] class weights: power={CLASS_WEIGHT_POWER}, '
              f'clamp=[{CLASS_WEIGHT_MIN}, {CLASS_WEIGHT_MAX}]')
        print(f'[v5] confusion loss: weight={CONFUSION_WEIGHT} '
              f'(artery idx={CONFUSION_ARTERY_IDX}, vein idx={CONFUSION_VEIN_IDX})')
        print(f'[v5] topo без изменений: weight={TOPO_WEIGHT}, '
              f'warmup={TOPO_WARMUP_EPOCHS}, ramp={TOPO_RAMP_EPOCHS}')

        self._ensure_scheduler(len(train_loader))
        self.history = []

        best_dice = -1.0
        best_state_dice: Optional[Dict[str, torch.Tensor]] = None
        band_topo = float("inf")
        band_dice = -1.0
        band_state: Optional[Dict[str, torch.Tensor]] = None

        pre_topo_dice: Optional[float] = None
        adaptive_applied = False

        start = time.time()
        for epoch in range(EPOCHS_COUNT):
            self.model.train()
            train_loss = 0.0
            ep_seg, ep_conf, ep_topo, ep_w = 0.0, 0.0, 0.0, 0.0
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
                ep_conf += float(info["loss_confusion"])
                ep_topo += float(info["loss_topo_raw"])
                ep_w = float(info["topo_weight"])
                n_steps += 1

            n_steps = max(n_steps, 1)
            train_loss /= n_steps
            ep_seg /= n_steps
            ep_conf /= n_steps
            ep_topo /= n_steps

            current_lr = self.optimizer.param_groups[0]["lr"]
            approx_time_left = (time.time() - start) / (epoch + 1) * (EPOCHS_COUNT - epoch - 1)
            time_left_str = utils.beautify_time_left(approx_time_left)

            record = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": train_loss,
                "train_seg_loss": ep_seg,
                "train_confusion_loss": ep_conf,
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

                if epoch == TOPO_WARMUP_EPOCHS - 1:
                    pre_topo_dice = v_dice
                    print(f"[topo] Запомнил Dice перед topo: {pre_topo_dice:.4f}")

                if (ADAPTIVE_TOPO_WEIGHT and not adaptive_applied
                    and epoch == ADAPTIVE_CHECK_EPOCH - 1 and pre_topo_dice is not None):
                    dice_drop = pre_topo_dice - v_dice
                    if dice_drop > DICE_DROP_THRESHOLD:
                        old_weight = self.criterion.topo.weight
                        self.criterion.topo.weight *= 0.5
                        print(f"[topo] Dice упал на {dice_drop:.3f} "
                              f"(с {pre_topo_dice:.4f} до {v_dice:.4f}). "
                              f"Уменьшаю topo_weight: {old_weight:.3f} -> {self.criterion.topo.weight:.3f}")
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
                  f"conf {ep_conf:.4f} + {ep_w:.2f}·topo {ep_topo:.4f})"
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
                "confusion_weight": CONFUSION_WEIGHT,
                "class_weight_power": CLASS_WEIGHT_POWER,
                "version": "v5",
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
        return 'UNET-WCD-TOPO-v5'

    @staticmethod
    def get_val_viz_path() -> pathlib.Path:
        return pathlib.Path('UNETPP_WCD/val_viz_patches')