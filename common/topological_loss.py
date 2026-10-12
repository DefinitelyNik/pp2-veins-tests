# common/topological_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalVesselLoss(nn.Module):
    """
    Нормализованный Топологический Лосс.
    Деление на (H * W) приводит значение к диапазону [0, 1],
    что делает его сопоставимым с Dice и CE Loss.
    """
    def __init__(self, num_classes=5, vessel_classes=None, num_thresholds=3):
        super().__init__()
        self.num_classes = num_classes
        self.vessel_classes = vessel_classes if vessel_classes else list(range(1, num_classes))
        self.num_thresholds = num_thresholds

    def soft_euler_characteristic(self, p):
        B, H, W = p.shape
        area = H * W  # Нормализация по площади

        V = torch.sum(p, dim=(1, 2))
        E_h = torch.sum(p[:, :, :-1] * p[:, :, 1:], dim=(1, 2))
        E_v = torch.sum(p[:, :-1, :] * p[:, 1:, :], dim=(1, 2))
        E = E_h + E_v
        F = torch.sum(p[:, :-1, :-1] * p[:, :-1, 1:] * p[:, 1:, :-1] * p[:, 1:, 1:], dim=(1, 2))

        chi = V - E + F
        return chi / area

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)

        if targets.dim() == 3:
            if targets.dtype in (torch.long, torch.int, torch.int64, torch.int32):
                targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
            else:
                targets_one_hot = targets
        else:
            targets_one_hot = targets.float()

        topo_loss = 0.0
        thresholds = torch.linspace(0.2, 0.8, self.num_thresholds, device=logits.device)

        for t in thresholds:
            p_pred = torch.sigmoid((probs - t) * 5.0)
            p_gt = torch.sigmoid((targets_one_hot - t) * 5.0)

            for c in self.vessel_classes:
                chi_pred = self.soft_euler_characteristic(p_pred[:, c, :, :])
                chi_gt = self.soft_euler_characteristic(p_gt[:, c, :, :])
                topo_loss += torch.mean(torch.abs(chi_pred - chi_gt))

        return topo_loss / (self.num_thresholds * len(self.vessel_classes))