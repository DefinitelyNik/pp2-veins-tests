"""
Топологический лосс на основе Soft Euler Characteristic из статьи HyperTopo-Adapters.

Алгоритм из статьи (Algorithm 1):
1. V(P) = Pi,j · Pi+1,j · Pi,j+1 · Pi+1,j+1 (Soft Vertices)
2. Eh(P) = Pi,j · Pi,j+1 (Horizontal Edges)
3. Ev(P) = Pi,j · Pi+1,j (Vertical Edges)
4. F(P) = Pi,j (Faces/Pixels)
5. χsoft(P) = Σ(F(P) - Eh(P) - Ev(P) + V(P))
6. Loss = ||χsoft(P) - χgt||²

Euler characteristic для 2D: χ = β0 - β1 (components - holes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SoftEulerCharacteristicLoss(nn.Module):
    """
    Soft Euler Characteristic Loss для multiclass segmentation.

    Вычисляет топологический лосс на основе эйлеровой характеристики,
    что помогает сохранять топологическую структуру (число компонент β0 и дыр β1).
    Использует дифференцируемую замену бинарных значений на вероятности [0, 1],
    что позволяет градиентам протекать в веса адаптера (как в статье).

    Args:
        from_logits: Если True, применяет softmax к входным логитам
        temperature: Температура для softmax (по умолчанию 1.0)
        use_squared_loss: Если True, используется L2 loss, иначе L1
        per_class: Если True, лосс вычисляется для каждого класса отдельно и усредняется
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
        """
        Вычисляет soft Euler characteristic для probability map одной маски.

        Реализация по Algorithm 1 из статьи:
        χ = Σ(F - Eh - Ev + V), где:
        - F: faces (пиксели)
        - Eh: horizontal edges
        - Ev: vertical edges
        - V: vertices (углы 2x2)

        Args:
            prob_map: Tensor shape [B, H, W] с вероятностями [0, 1]

        Returns:
            Tensor shape [B] со значениями χ для каждого изображения в батче
        """
        # prob_map: [B, H, W]
        B, H, W = prob_map.shape

        # F(P) = Pi,j (Faces/Pixels)
        # Для pixel grid: каждый пиксель = face
        faces = prob_map  # [B, H, W]
        faces_sum = faces.sum(dim=(1, 2))  # [B]

        # Eh(P) = Pi,j · Pi,j+1 (Horizontal Edges)
        # Ребро между соседними пикселями по горизонтали
        # shape: [B, H, W-1]
        horizontal_edges = prob_map[:, :, :-1] * prob_map[:, :, 1:]
        eh_sum = horizontal_edges.sum(dim=(1, 2))  # [B]

        # Ev(P) = Pi,j · Pi+1,j (Vertical Edges)
        # Ребро между соседними пикселями по вертикали
        # shape: [B, H-1, W]
        vertical_edges = prob_map[:, :-1, :] * prob_map[:, 1:, :]
        ev_sum = vertical_edges.sum(dim=(1, 2))  # [B]

        # V(P) = Pi,j · Pi+1,j · Pi,j+1 · Pi+1,j+1 (Soft Vertices)
        # Вершина = угол 2x2 квадрата
        # shape: [B, H-1, W-1]
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
        """
        Вычисляет Soft Euler Characteristic Loss.

        Args:
            logits: Tensor shape [B, C, H, W] - выходные логиты модели
            targets: Tensor shape [B, H, W] - ground truth маски с индексами классов [0, C-1]

        Returns:
            Scalar loss value
        """
        if self.from_logits:
            # Применяем softmax с temperature для получения вероятностей
            probs = F.softmax(logits / self.temperature, dim=1)  # [B, C, H, W]
        else:
            probs = logits

        B, C, H, W = probs.shape
        device = probs.device

        if self.per_class:
            # Вычисляем χ для каждого класса отдельно и усредняем
            total_loss = torch.tensor(0.0, device=device)

            for c in range(C):
                # Predicted probability map для класса c
                pred_prob = probs[:, c, :, :]  # [B, H, W]
                # Ground truth binary mask для класса c
                gt_binary = (targets == c).float()  # [B, H, W]

                # Вычисляем χ для prediction и ground truth
                chi_pred = self._compute_euler_characteristic(pred_prob)  # [B]
                chi_gt = self._compute_euler_characteristic(gt_binary)  # [B]

                # Loss = ||χ_pred - χ_gt||² (усредняем по батчу)
                if self.use_squared_loss:
                    loss_c = torch.mean((chi_pred - chi_gt) ** 2)
                else:
                    loss_c = torch.mean(torch.abs(chi_pred - chi_gt))

                total_loss = total_loss + loss_c

            # Усредняем по классам
            total_loss = total_loss / C
        else:
            # Альтернативный вариант: вычисляем χ для всей карты сразу (используя argmax)
            pred_argmax = torch.argmax(logits, dim=1).float()  # [B, H, W]
            chi_pred = self._compute_euler_characteristic(pred_argmax)
            chi_gt = self._compute_euler_characteristic(targets.float())

            if self.use_squared_loss:
                total_loss = torch.mean((chi_pred - chi_gt) ** 2)
            else:
                total_loss = torch.mean(torch.abs(chi_pred - chi_gt))

        return total_loss


class WeightedCEDiceTopoLoss(nn.Module):
    """
    Комбинированный лосс: Weighted CrossEntropy + Dice + Topological (Soft Euler Characteristic).

    Объединяет:
    - CrossEntropy для пиксельной классификации
    - Dice для работы с дисбалансом классов
    - Soft Euler Characteristic для сохранения топологической структуры (β0 и β1)

    Как в статье, добавляем warmup: первые N эпох топологический лосс отключен,
    чтобы позволить сегментационной маске стабилизироваться (см. Figure 8 статьи).

    Args:
        class_weights: Веса классов для CrossEntropy
        ce_coef: Коэффициент для CrossEntropy loss
        dice_coef: Коэффициент для Dice loss
        topo_coef: Коэффициент для топологического лосса (в статье PH weight)
        topo_temperature: Температура для softmax в топологическом лоссе
        topo_warmup_epochs: Количество эпох без топологического лосса (warmup)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ce_coef: float = 1.0,
        dice_coef: float = 1.0,
        topo_coef: float = 0.1,
        topo_temperature: float = 1.0,
        topo_warmup_epochs: int = 10,
    ):
        super().__init__()

        # Импортируем здесь чтобы избежать circular imports
        import segmentation_models_pytorch as smp

        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)
        self.topo = SoftEulerCharacteristicLoss(
            from_logits=True,
            temperature=topo_temperature,
            per_class=True,
            use_squared_loss=True,
        )

        self.ce_coef = ce_coef
        self.dice_coef = dice_coef
        self.topo_coef = topo_coef
        self.topo_warmup_epochs = topo_warmup_epochs
        self.current_epoch = 0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет комбинированный лосс.

        Args:
            logits: Tensor shape [B, C, H, W] - выходные логиты модели
            targets: Tensor shape [B, H, W] - ground truth маски

        Returns:
            Scalar loss value
        """
        # Базовые лоссы
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = self.ce_coef * ce_loss + self.dice_coef * dice_loss

        # Добавляем топологический лосс ПОСЛЕ warmup (как в статье Figure 8)
        if self.current_epoch >= self.topo_warmup_epochs:
            topo_loss = self.topo(logits, targets)
            total_loss = total_loss + self.topo_coef * topo_loss

        return total_loss

    def set_epoch(self, epoch: int):
        """Устанавливает текущую эпоху для warmup логики."""
        self.current_epoch = epoch

    def get_loss_components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Возвращает значения отдельных компонентов лосса для логирования.

        Returns:
            Dictionary с ключами 'ce', 'dice', 'topo', 'total'
        """
        with torch.no_grad():
            ce_loss = self.ce(logits, targets)
            dice_loss = self.dice(logits, targets)
            topo_loss = self.topo(logits, targets)

            total = self.ce_coef * ce_loss + self.dice_coef * dice_loss
            if self.current_epoch >= self.topo_warmup_epochs:
                total = total + self.topo_coef * topo_loss

            return {
                'ce': ce_loss.item(),
                'dice': dice_loss.item(),
                'topo': topo_loss.item(),
                'total': total.item(),
                'epoch': self.current_epoch,
                'topo_active': self.current_epoch >= self.topo_warmup_epochs,
            }