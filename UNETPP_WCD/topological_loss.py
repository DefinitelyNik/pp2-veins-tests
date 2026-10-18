"""
Топологический лосс на основе Soft Euler Characteristic (HyperTopo-Adapters).
Оптимизирован для 50 эпох обучения и сильного дисбаланса классов.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import segmentation_models_pytorch as smp


class FocalLoss(nn.Module):
    """
    Focal Loss для multiclass segmentation.
    Подавляет лёгкие примеры (фон) и фокусируется на редких классах.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        probs = F.softmax(logits, dim=1)
        targets_expanded = targets.unsqueeze(1)
        p_t = torch.gather(probs, 1, targets_expanded).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class WeightedMulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice с весами по классам.
    Редкие классы получают больший штраф.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        from_logits: bool = True,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.from_logits = from_logits
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits

        B, C, H, W = probs.shape
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(C):
            pred_flat = probs[:, c, :, :].view(B, -1)
            target_flat = targets_one_hot[:, c, :, :].view(B, -1)
            intersection = (pred_flat * target_flat).sum(dim=1)
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(1.0 - dice.mean())

        dice_per_class = torch.stack(dice_per_class)

        if self.class_weights is not None:
            w = self.class_weights.to(dice_per_class.device)
            return (dice_per_class * w).sum() / w.sum()
        return dice_per_class.mean()


class SoftEulerCharacteristicLoss(nn.Module):
    """
    Soft Euler Characteristic Loss (Algorithm 1 из статьи).
    χ = Σ(F − Eh − Ev + V), где:
      F  = Pi,j                    (faces / пиксели)
      Eh = Pi,j · Pi,j+1           (horizontal edges)
      Ev = Pi,j · Pi+1,j           (vertical edges)
      V  = Pi,j · Pi+1,j · Pi,j+1 · Pi+1,j+1  (vertices)
    Loss = ||χ_pred − χ_gt||²
    """

    def __init__(self, from_logits=True, temperature=1.0, per_class=True):
        super().__init__()
        self.from_logits = from_logits
        self.temperature = temperature
        self.per_class = per_class

    def _compute_chi(self, prob_map: torch.Tensor) -> torch.Tensor:
        faces = prob_map.sum(dim=(1, 2))
        eh = (prob_map[:, :, :-1] * prob_map[:, :, 1:]).sum(dim=(1, 2))
        ev = (prob_map[:, :-1, :] * prob_map[:, 1:, :]).sum(dim=(1, 2))
        v = (
            prob_map[:, :-1, :-1] * prob_map[:, 1:, :-1]
            * prob_map[:, :-1, 1:] * prob_map[:, 1:, 1:]
        ).sum(dim=(1, 2))
        return faces - eh - ev + v

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            probs = F.softmax(logits / self.temperature, dim=1)
        else:
            probs = logits

        B, C, H, W = probs.shape
        total = torch.tensor(0.0, device=probs.device)

        for c in range(C):
            chi_pred = self._compute_chi(probs[:, c])
            chi_gt = self._compute_chi((targets == c).float())
            total = total + torch.mean((chi_pred - chi_gt) ** 2)

        return total / C


class WeightedFocalDiceTopoLoss(nn.Module):
    """
    Focal + Weighted Dice + Soft Euler Characteristic.
    Оптимизирован для 50 эпох:
      • эпохи 1–5   → только Focal + Dice (стабилизация сегментации)
      • эпохи 6–12  → плавное включение Topo (ramp-up)
      • эпохи 13–50 → полный лосс
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_coef: float = 1.0,
        dice_coef: float = 2.0,
        topo_coef: float = 0.001,
        topo_temperature: float = 1.0,
        topo_warmup_epochs: int = 5,
        topo_rampup_epochs: int = 7,
    ):
        super().__init__()
        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice = WeightedMulticlassDiceLoss(class_weights=class_weights)
        self.topo = SoftEulerCharacteristicLoss(
            from_logits=True, temperature=topo_temperature, per_class=True,
        )
        self.focal_coef = focal_coef
        self.dice_coef = dice_coef
        self.max_topo_coef = topo_coef
        self.topo_warmup_epochs = topo_warmup_epochs
        self.topo_rampup_epochs = topo_rampup_epochs
        self.current_epoch = 0

    def _topo_weight(self) -> float:
        if self.current_epoch < self.topo_warmup_epochs:
            return 0.0
        t = self.current_epoch - self.topo_warmup_epochs
        if t >= self.topo_rampup_epochs:
            return self.max_topo_coef
        return self.max_topo_coef * (t / self.topo_rampup_epochs)

    def forward(self, logits, targets):
        loss = self.focal_coef * self.focal(logits, targets) \
             + self.dice_coef * self.dice(logits, targets)
        w = self._topo_weight()
        if w > 0:
            loss = loss + w * self.topo(logits, targets)
        return loss

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_loss_components(self, logits, targets):
        with torch.no_grad():
            f = self.focal(logits, targets)
            d = self.dice(logits, targets)
            t = self.topo(logits, targets)
            w = self._topo_weight()
            total = self.focal_coef * f + self.dice_coef * d
            if w > 0:
                total = total + w * t
            return {
                'focal': f.item(), 'dice': d.item(),
                'topo': t.item(), 'topo_coef': w, 'total': total.item(),
            }