# common/topological_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalVesselLoss(nn.Module):
    """
    Дифференцируемый Топологический Лосс (Soft Euler Characteristic).
    Штрафует модель за нарушение связности сосудов (разрывы и лишние циклы).
    """
    def __init__(self, num_classes=5, vessel_classes=None, num_thresholds=5):
        super().__init__()
        self.num_classes = num_classes
        # По умолчанию применяем ко всем классам, кроме фона (0)
        self.vessel_classes = vessel_classes if vessel_classes else list(range(1, num_classes))
        self.num_thresholds = num_thresholds

    def soft_euler_characteristic(self, p):
        """
        Вычисляет "мягкую" Эйлерову характеристику для 2D карты вероятностей.
        Chi = V - E + F (Вершины - Ребра + Грани)
        """
        # V: Вершины (сумма вероятностей всех пикселей)
        V = torch.sum(p, dim=(1, 2))

        # E: Ребра (горизонтальные и вертикальные связи между соседями)
        E_h = torch.sum(p[:, :, :-1] * p[:, :, 1:], dim=(1, 2))
        E_v = torch.sum(p[:, :-1, :] * p[:, 1:, :], dim=(1, 2))
        E = E_h + E_v

        # F: Грани (квадраты 2x2)
        F = torch.sum(p[:, :-1, :-1] * p[:, :-1, 1:] * p[:, 1:, :-1] * p[:, 1:, 1:], dim=(1, 2))

        return V - E + F

    def forward(self, logits, targets):
        # logits: (B, C, H, W)
        probs = torch.softmax(logits, dim=1)

        # Преобразуем targets в one-hot формат, если переданы индексы классов
        if targets.dim() == 3:
            if targets.dtype in (torch.long, torch.int, torch.int64, torch.int32):
                targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = targets
        else:
            targets_one_hot = targets.float()

        topo_loss = 0.0
        # Интегрирование по порогам (Euler Characteristic Curve) для устойчивости
        thresholds = torch.linspace(0.1, 0.9, self.num_thresholds, device=logits.device)

        for t in thresholds:
            # Мягкое пороговое значение (differentiable step) с температурой 10.0
            p_pred = torch.sigmoid((probs - t) * 10.0)
            p_gt = torch.sigmoid((targets_one_hot - t) * 10.0)

            for c in self.vessel_classes:
                chi_pred = self.soft_euler_characteristic(p_pred[:, c, :, :])
                chi_gt = self.soft_euler_characteristic(p_gt[:, c, :, :])

                # L1 дистанция между топологическими инвариантами
                topo_loss += torch.mean(torch.abs(chi_pred - chi_gt))

        # Нормализация на количество порогов и классов
        return topo_loss / (self.num_thresholds * len(self.vessel_classes))