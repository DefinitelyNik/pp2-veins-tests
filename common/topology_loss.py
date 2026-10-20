# common/topology_loss.py
"""
Топологический лосс на основе дифференцируемой мягкой эйлеровой характеристики
(Soft Euler Characteristic, SEC).

Реализация по мотивам:
    Ch. W. Ndubuisi, "HyperTopo-Adapters: Geometry- and Topology-Aware
    Segmentation of Leaf Lesions on Frozen Encoders", arXiv:2601.06067 (2025).
    Algorithm 1 "Soft Euler Characteristic Loss", Appendix A.2, Fig. 8 (warmup).

Адаптация под многоклассовую сегментацию сосудов сетчатки (5 классов):
    0 background, 1 arteries, 2 veins, 3 crossings, 4 capillaries

--------------------------------------------------------------------------
ИСПРАВЛЕНИЯ v2 (после анализа коллапса в логах)
--------------------------------------------------------------------------
1. exclude_background=True (ПО УМОЛЧАНИЮ): фон исключён из per-class topo loss.
   Причина: χ_soft для фона имеет систематическое смещение ~−0.026/pixel при
   q=0.97, что даёт ~−1700 на патч 256×256 при χ_gt≈0. Это создаёт огромный
   штраф, который доминирует в лоссе и тянет модель в коллапс.

2. denom = sqrt(1 + |χ_gt|) вместо (1 + |χ_gt|): более мягкая нормализация,
   чтобы классы с большой топологией (capillaries: |χ|~50) не были "прощены"
   за большие абсолютные ошибки, но и не доминировали.

3. gradient_mode="ste" по умолчанию: значение и градиент вычисляются в одной
   точке (straight-through estimator). Устраняет рассинхронизацию между
   χ_hard (значение) и dχ_soft/dp (градиент), которая была причиной коллапса.

4. absent_class_scale=0.0 по умолчанию: классы, отсутствующие в GT патча,
   полностью игнорируются в topo loss. Иначе модель учится "не предсказывать
   редкие классы" (crossings есть только на ~10% патчей).
--------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as tF

__all__ = [
    "soft_euler_characteristic",
    "euler_characteristic_of_index_map",
    "SoftEulerCharacteristicLoss",
]


# ===========================================================================
# Ядро: дифференцируемая эйлерова характеристика
# ===========================================================================
def soft_euler_characteristic(p: torch.Tensor) -> torch.Tensor:
    """
    Дифференцируемая эйлерова характеристика χ = V − E + F = β0 − β1.

    Локальная 2×2 форма (идентична прямой, но численно устойчивее):
        L(a,b,c,d) = ¼(a+b+c+d) − ½(ab+cd+ac+bd) + abcd
        χ = Σ_окна L,   окна бегут по паддингованной нулями карте
    """
    pp = tF.pad(p, (1, 1, 1, 1), mode="constant", value=0.0)

    a = pp[..., :-1, :-1]  # top-left
    b = pp[..., :-1, 1:]  # top-right
    c = pp[..., 1:, :-1]  # bottom-left
    d = pp[..., 1:, 1:]  # bottom-right

    ab = a * b
    cd = c * d
    ac = a * c
    bd = b * d

    local = 0.25 * (a + b + c + d) - 0.5 * (ab + cd + ac + bd) + ab * cd
    return local.sum(dim=(-2, -1))


def euler_characteristic_of_index_map(
        targets: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Точная (бинарная) χ для каждого класса из карты индексов."""
    with torch.no_grad():
        t = targets.clamp(min=0, max=num_classes - 1)
        oh = tF.one_hot(t, num_classes)
        oh = oh.permute(0, 3, 1, 2).to(torch.float32)
        return soft_euler_characteristic(oh)


def _total_variation(p: torch.Tensor) -> torch.Tensor:
    """TV-регуляризация для стабильности обучения."""
    dh = (p[..., 1:, :] - p[..., :-1, :]).abs()
    dw = (p[..., :, 1:] - p[..., :, :-1]).abs()
    return dh.mean() + dw.mean()


# ===========================================================================
# Лосс
# ===========================================================================
class SoftEulerCharacteristicLoss(nn.Module):
    """
    Многоклассовый топологический лосс: |χ_pred − χ_gt| по каждому классу
    + по объединению всего переднего плана.

    КЛЮЧЕВЫЕ ПАРАМЕТРЫ ДЛЯ БАЛАНСА КЛАССОВ
    --------------------------------------
    exclude_background   — исключить фон из per-class topo loss. ПО УМОЛЧАНИЮ
                           True. Фон имеет систематическое смещение χ_soft и
                           доминирует в лоссе, вызывая коллапс.
    weight_temperature   — степень для class_weights. 0.3 (default) даёт
                           умеренное усиление редких классов (~4× для crossings).
    absent_class_scale   — множитель веса для классов, отсутствующих в GT.
                           0.0 (default) = полностью игнорировать. Иначе модель
                           учится не предсказывать редкие классы.
    per-sample ренормировка — каждый патч даёт одинаковый вклад.

    ЧИСЛЕННАЯ УСТОЙЧИВОСТЬ
    ----------------------
    value_mode="soft"    — значение и градиент через мягкие вероятности (согласовано).
                           Рекомендуется с temperature<1 для острого softmax.
    value_mode="hard"    — значение на бинарной карте (без смещения), требует
                           gradient_mode="ste" для согласованного градиента.
    gradient_mode="soft" — градиент через мягкие вероятности (по умолчанию).
    gradient_mode="ste"  — straight-through: значение hard, градиент soft в
                           точке hard (устраняет рассинхронизацию).
    rel_clamp            — ограничение относительной ошибки (защита от взрыва).

    WARMUP (Fig. 8 статьи)
    ----------------------
    weight_at_epoch(epoch) == 0 первые `warmup_epochs`, затем линейный рост.
    """

    def __init__(
            self,
            num_classes: int,
            class_weights: Optional[torch.Tensor] = None,
            weight: float = 0.5,
            warmup_epochs: int = 10,
            ramp_epochs: int = 5,
            value_mode: str = "soft",  # v2: soft по умолчанию (согласовано)
            gradient_mode: str = "ste",  # v2: ste по умолчанию (устойчиво)
            temperature: float = 1.0,
            weight_temperature: float = 0.3,  # v2: 0.3 вместо 0.5 (умереннее)
            absent_class_scale: float = 0.0,  # v2: 0.0 вместо 0.5 (не штраф отсутствующих)
            rel_clamp: float = 1000.0,  # v2: 1000 вместо 5 (не обрезать градиент)
            reduction: str = "smooth_l1",
            include_union: bool = True,
            union_weight: float = 1.0,
            tv_weight: float = 0.0,
            background_index: int = 0,
            exclude_background: bool = True,  # v2: исключить фон из topo loss
            eps: float = 1e-8,
    ):
        super().__init__()
        assert value_mode in ("hard", "soft"), value_mode
        assert gradient_mode in ("soft", "ste"), gradient_mode
        assert reduction in ("smooth_l1", "l2"), reduction

        self.num_classes = int(num_classes)
        self.weight = float(weight)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.value_mode = value_mode
        self.gradient_mode = gradient_mode
        self.temperature = float(temperature)
        self.weight_temperature = float(weight_temperature)
        self.absent_class_scale = float(absent_class_scale)
        self.rel_clamp = float(rel_clamp)
        self.reduction = reduction
        self.include_union = bool(include_union)
        self.union_weight = float(union_weight)
        self.tv_weight = float(tv_weight)
        self.background_index = int(background_index)
        self.exclude_background = bool(exclude_background)
        self.eps = float(eps)

        # --- class-balanced веса для топологического члена ---
        if class_weights is None:
            w = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            w = class_weights.detach().to(torch.float32).clamp(min=0.0)
        w = w.pow(self.weight_temperature)
        w = w / w.mean().clamp(min=self.eps)
        self.register_buffer("class_weights", w)

        self._last_info: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def weight_at_epoch(self, epoch: int) -> float:
        if self.weight <= 0.0:
            return 0.0
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return self.weight
        t = (epoch - self.warmup_epochs + 1) / float(self.ramp_epochs)
        return self.weight * float(min(1.0, t))

    # ------------------------------------------------------------------
    def _reduce(self, rel: torch.Tensor) -> torch.Tensor:
        rel = rel.clamp(-self.rel_clamp, self.rel_clamp)
        if self.reduction == "l2":
            return rel.pow(2)
        return tF.smooth_l1_loss(rel, torch.zeros_like(rel), beta=1.0, reduction="none")

    @staticmethod
    def _hard_onehot(probs: torch.Tensor) -> torch.Tensor:
        idx = probs.argmax(dim=1)
        oh = tF.one_hot(idx, probs.shape[1])
        return oh.permute(0, 3, 1, 2).to(probs.dtype).contiguous()

    # ------------------------------------------------------------------
    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            return_info: bool = False,
    ):
        C = self.num_classes
        probs = torch.softmax(logits / self.temperature, dim=1)

        bg = self.background_index
        fg_soft = 1.0 - probs[:, bg:bg + 1]

        p_soft_all = torch.cat([probs, fg_soft], dim=1)
        with torch.no_grad():
            oh_hard = self._hard_onehot(probs)
            fg_hard = (oh_hard.argmax(dim=1) != bg).to(probs.dtype).unsqueeze(1)
            p_hard_all = torch.cat([oh_hard, fg_hard], dim=1)

        # --- χ предсказания ---
        if self.value_mode == "soft":
            chi_pred_all = soft_euler_characteristic(p_soft_all)
            chi_hard_all = None
        elif self.gradient_mode == "ste":
            # STE: значение = hard, градиент течёт через soft
            p_ste = p_hard_all + (p_soft_all - p_soft_all.detach())
            chi_pred_all = soft_euler_characteristic(p_ste)
            chi_hard_all = None
        else:
            # value_mode="hard", gradient_mode="soft" (НЕ рекомендуется)
            chi_soft_all = soft_euler_characteristic(p_soft_all)
            with torch.no_grad():
                chi_hard_all = soft_euler_characteristic(p_hard_all)
            chi_pred_all = chi_hard_all + (chi_soft_all - chi_soft_all.detach())

        # --- χ и площади ground truth ---
        with torch.no_grad():
            t = targets.clamp(min=0, max=C - 1)
            tgt_oh = tF.one_hot(t, C).permute(0, 3, 1, 2).to(torch.float32)
            tgt_fg = (t != bg).to(torch.float32).unsqueeze(1)
            tgt_all = torch.cat([tgt_oh, tgt_fg], dim=1)
            chi_gt_all = soft_euler_characteristic(tgt_all)
            area_gt = tgt_all.sum(dim=(2, 3))

        # ================== per-class член ==================
        chi_pred_c = chi_pred_all[:, :C]
        chi_gt_c = chi_gt_all[:, :C]

        # v2: denom = sqrt(1 + |χ_gt|) для более мягкой нормализации
        denom_c = (1.0 + chi_gt_c.abs()).sqrt()
        rel_c = (chi_pred_c - chi_gt_c) / denom_c
        per_class = self._reduce(rel_c)

        # v2: исключить фон из topo loss (главное исправление!)
        if self.exclude_background and bg < C:
            per_class[:, bg] = 0.0

        # presence-aware веса + per-sample ренормировка
        present = (area_gt[:, :C] > 0).to(per_class.dtype)
        presence_mul = present + self.absent_class_scale * (1.0 - present)

        # v2: если exclude_background, обнулить вес для фона
        w_class = self.class_weights.to(per_class.dtype).clone()
        if self.exclude_background and bg < C:
            w_class[bg] = 0.0

        w_eff = w_class.unsqueeze(0) * presence_mul
        w_eff = w_eff / w_eff.mean(dim=1, keepdim=True).clamp(min=self.eps)

        loss_cls = (per_class * w_eff).mean()

        # ================== член объединения (foreground) ==================
        if self.include_union:
            chi_pred_u = chi_pred_all[:, C]
            chi_gt_u = chi_gt_all[:, C]
            denom_u = (1.0 + chi_gt_u.abs()).sqrt()
            rel_u = (chi_pred_u - chi_gt_u) / denom_u
            loss_u = self._reduce(rel_u).mean()
            total = (loss_cls + self.union_weight * loss_u) / (1.0 + self.union_weight)
        else:
            loss_u = torch.zeros((), device=logits.device, dtype=logits.dtype)
            total = loss_cls

        # ================== опциональный TV ==================
        if self.tv_weight > 0.0:
            tv = _total_variation(p_soft_all)
            total = total + self.tv_weight * tv
        else:
            tv = torch.zeros((), device=logits.device, dtype=logits.dtype)

        # ================== диагностика ==================
        with torch.no_grad():
            chi_val = chi_hard_all[:, :C] if chi_hard_all is not None else chi_pred_c.detach()
            chi_fg_val = (
                chi_hard_all[:, C] if chi_hard_all is not None else chi_pred_all[:, C].detach()
            )
            info = {
                "topo/loss": float(total.detach()),
                "topo/loss_cls": float(loss_cls.detach()),
                "topo/loss_union": float(loss_u.detach()),
                "topo/tv": float(tv.detach()),
                "topo/dchi_abs_cls": float(
                    (chi_val - chi_gt_c).abs().mean()
                ),
                "topo/dchi_abs_union": float((chi_fg_val - chi_gt_u).abs().mean()),
                "topo/rel_err_cls": float(rel_c.abs().mean()),
            }
        self._last_info = info

        if return_info:
            return total, info
        return total

    @property
    def last_info(self) -> Dict[str, float]:
        return dict(self._last_info)

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, weight={self.weight}, "
            f"warmup={self.warmup_epochs}, ramp={self.ramp_epochs}, "
            f"value_mode={self.value_mode}, gradient_mode={self.gradient_mode}, "
            f"exclude_background={self.exclude_background}, "
            f"weight_temperature={self.weight_temperature}, "
            f"absent_class_scale={self.absent_class_scale}, "
            f"include_union={self.include_union}, union_weight={self.union_weight}"
        )