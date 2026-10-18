"""
Топологический лосс с Focal Loss и улучшенной балансировкой классов.

Ключевые улучшения:
1. Focal Loss вместо CE для борьбы с class imbalance
2. Weighted Multiclass Dice для усиления редких классов
3. Агрессивные class weights с effective number of samples
4. Плавное включение топологического лосса
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import segmentation_models_pytorch as smp


class FocalLoss(nn.Module):
    """
    Focal Loss для multiclass segmentation.
    Подавляет легкие примеры (background) и фокусируется на сложных (редкие классы).
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W] - raw model outputs
            targets: [B, H, W] - class indices
        """
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')

        # Get probabilities for true classes
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        targets_expanded = targets.unsqueeze(1)  # [B, 1, H, W]
        p_t = torch.gather(probs, 1, targets_expanded).squeeze(1)  # [B, H, W]

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply focal weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedMulticlassDiceLoss(nn.Module):
    """
    Multiclass Dice Loss с весами по классам.
    Редкие классы получают больший штраф при ошибках.
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
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W]
        """
        if self.from_logits:
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        else:
            probs = logits

        B, C, H, W = probs.shape

        # Convert targets to one-hot: [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        # Compute Dice per class
        dice_per_class = []
        for c in range(C):
            pred_c = probs[:, c, :, :]  # [B, H, W]
            target_c = targets_one_hot[:, c, :, :]  # [B, H, W]

            # Flatten spatial dimensions
            pred_flat = pred_c.view(B, -1)  # [B, H*W]
            target_flat = target_c.view(B, -1)  # [B, H*W]

            # Dice coefficient
            intersection = (pred_flat * target_flat).sum(dim=1)  # [B]
            union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)  # [B]

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # [B]
            dice_loss = 1.0 - dice.mean()  # Scalar

            dice_per_class.append(dice_loss)

        dice_per_class = torch.stack(dice_per_class)  # [C]

        # Apply class weights
        if self.class_weights is not None:
            weighted_dice = (dice_per_class * self.class_weights.to(dice_per_class.device)).sum()
            weighted_dice = weighted_dice / self.class_weights.sum()
        else:
            weighted_dice = dice_per_class.mean()

        return weighted_dice


class SoftEulerCharacteristicLoss(nn.Module):
    """
    Soft Euler Characteristic Loss для multiclass segmentation.
    (Без изменений из предыдущей версии)
    """

    def __init__(
        self,
        from_logits: bool = True,
        temperature: float = 1.0,
        use_squared_loss: bool = True,
        per_class: bool = True,
    ):
        super().__init__()
        self.from_logits = from_logits
        self.temperature = temperature
        self.use_squared_loss = use_squared_loss
        self.per_class = per_class

    def _compute_euler_characteristic(self, prob_map: torch.Tensor) -> torch.Tensor:
        B, H, W = prob_map.shape

        # F(P) = Pi,j (Faces/Pixels)
        faces_sum = prob_map.sum(dim=(1, 2))  # [B]

        # Eh(P) = Pi,j · Pi,j+1 (Horizontal Edges)
        horizontal_edges = prob_map[:, :, :-1] * prob_map[:, :, 1:]
        eh_sum = horizontal_edges.sum(dim=(1, 2))  # [B]

        # Ev(P) = Pi,j · Pi+1,j (Vertical Edges)
        vertical_edges = prob_map[:, :-1, :] * prob_map[:, 1:, :]
        ev_sum = vertical_edges.sum(dim=(1, 2))  # [B]

        # V(P) = Pi,j · Pi+1,j · Pi,j+1 · Pi+1,j+1 (Soft Vertices)
        p00 = prob_map[:, :-1, :-1]
        p10 = prob_map[:, 1:, :-1]
        p01 = prob_map[:, :-1, 1:]
        p11 = prob_map[:, 1:, 1:]
        vertices = p00 * p10 * p01 * p11
        v_sum = vertices.sum(dim=(1, 2))  # [B]

        # χ = Σ(F - Eh - Ev + V)
        euler_char = faces_sum - eh_sum - ev_sum + v_sum  # [B]

        return euler_char

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if self.from_logits:
            probs = F.softmax(logits / self.temperature, dim=1)  # [B, C, H, W]
        else:
            probs = logits

        B, C, H, W = probs.shape
        device = probs.device

        if self.per_class:
            total_loss = torch.tensor(0.0, device=device)

            for c in range(C):
                pred_prob = probs[:, c, :, :]  # [B, H, W]
                gt_binary = (targets == c).float()  # [B, H, W]

                chi_pred = self._compute_euler_characteristic(pred_prob)  # [B]
                chi_gt = self._compute_euler_characteristic(gt_binary)  # [B]

                if self.use_squared_loss:
                    loss_c = torch.mean((chi_pred - chi_gt) ** 2)
                else:
                    loss_c = torch.mean(torch.abs(chi_pred - chi_gt))

                total_loss = total_loss + loss_c

            total_loss = total_loss / C
        else:
            pred_argmax = torch.argmax(logits, dim=1).float()  # [B, H, W]
            chi_pred = self._compute_euler_characteristic(pred_argmax)
            chi_gt = self._compute_euler_characteristic(targets.float())

            if self.use_squared_loss:
                total_loss = torch.mean((chi_pred - chi_gt) ** 2)
            else:
                total_loss = torch.mean(torch.abs(chi_pred - chi_gt))

        return total_loss


class WeightedFocalDiceTopoLoss(nn.Module):
    """
    Улучшенный комбинированный лосс:
    - Focal Loss (вместо CE) для борьбы с class imbalance
    - Weighted Multiclass Dice для усиления редких классов
    - Soft Euler Characteristic для топологии (с плавным включением)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        focal_coef: float = 1.0,
        dice_coef: float = 1.0,
        topo_coef: float = 0.005,          # <-- ЕЩЕ МЕНЬШЕ (было 0.01)
        topo_temperature: float = 1.0,
        topo_warmup_epochs: int = 15,       # <-- БОЛЬШЕ (было 10)
        topo_rampup_epochs: int = 15,       # <-- ПЛАВНЕЕ (было 10)
    ):
        super().__init__()

        self.focal = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.dice = WeightedMulticlassDiceLoss(class_weights=class_weights)
        self.topo = SoftEulerCharacteristicLoss(
            from_logits=True,
            temperature=topo_temperature,
            per_class=True,
            use_squared_loss=True,
        )

        self.focal_coef = focal_coef
        self.dice_coef = dice_coef
        self.max_topo_coef = topo_coef
        self.topo_warmup_epochs = topo_warmup_epochs
        self.topo_rampup_epochs = topo_rampup_epochs
        self.current_epoch = 0

    def _get_current_topo_coef(self) -> float:
        """Плавное линейное нарастание веса топологии после warmup."""
        if self.current_epoch < self.topo_warmup_epochs:
            return 0.0

        epochs_since_warmup = self.current_epoch - self.topo_warmup_epochs
        if epochs_since_warmup >= self.topo_rampup_epochs:
            return self.max_topo_coef

        return self.max_topo_coef * (epochs_since_warmup / self.topo_rampup_epochs)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        total_loss = self.focal_coef * focal_loss + self.dice_coef * dice_loss

        current_topo_coef = self._get_current_topo_coef()
        if current_topo_coef > 0:
            topo_loss = self.topo(logits, targets)
            total_loss = total_loss + current_topo_coef * topo_loss

        return total_loss

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def get_loss_components(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        with torch.no_grad():
            focal_loss = self.focal(logits, targets)
            dice_loss = self.dice(logits, targets)
            topo_loss = self.topo(logits, targets)

            current_topo_coef = self._get_current_topo_coef()
            total = self.focal_coef * focal_loss + self.dice_coef * dice_loss
            if current_topo_coef > 0:
                total = total + current_topo_coef * topo_loss

            return {
                'focal': focal_loss.item(),
                'dice': dice_loss.item(),
                'topo': topo_loss.item(),
                'topo_coef': current_topo_coef,
                'total': total.item(),
            }