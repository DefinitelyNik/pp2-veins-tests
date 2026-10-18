import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftEulerCharacteristicLoss(nn.Module):
    """
    Топологический лосс на основе Soft Euler Characteristic из статьи
    "HyperTopo-Adapters: Geometry- and Topology-Aware Segmentation of Leaf Lesions on Frozen Encoders"
    
    Вычисляет разницу между предсказанной и истинной эйлеровой характеристикой для каждого класса.
    χ = V - E + F (вершины - ребра + грани) = β0 - β1 (компоненты связности - дырки)
    
    Для multiclass сегментации применяется к вероятностям каждого класса отдельно.
    """
    
    def __init__(self, class_weights=None, from_logits=True):
        super().__init__()
        self.class_weights = class_weights
        self.from_logits = from_logits
    
    def _soft_euler_characteristic(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет soft Euler characteristic для одного канала (класса).
        
        Args:
            probs: Tensor shape [B, H, W] - вероятности для одного класса
            
        Returns:
            Tensor shape [B] - значение χ для каждого элемента батча
        """
        # Faces (F) - просто сумма всех пикселей
        F = probs.sum(dim=(1, 2))
        
        # Horizontal Edges (Eh) - Pi,j * Pi,j+1
        Eh = (probs[:, :, :-1] * probs[:, :, 1:]).sum(dim=(1, 2))
        
        # Vertical Edges (Ev) - Pi,j * Pi+1,j
        Ev = (probs[:, :-1, :] * probs[:, 1:, :]).sum(dim=(1, 2))
        
        # Vertices (V) - Pi,j * Pi+1,j * Pi,j+1 * Pi+1,j+1
        V = (probs[:, :-1, :-1] * probs[:, 1:, :-1] * 
             probs[:, :-1, 1:] * probs[:, 1:, 1:]).sum(dim=(1, 2))
        
        # χ = F - Eh - Ev + V
        chi = F - Eh - Ev + V
        
        return chi
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, epoch: int = 0, warmup_epochs: int = 10) -> torch.Tensor:
        """
        Args:
            logits: Tensor shape [B, C, H, W] - logits от модели
            targets: Tensor shape [B, H, W] - ground truth labels (class indices)
            epoch: Current training epoch
            warmup_epochs: Number of epochs to skip before applying topology loss
            
        Returns:
            Scalar loss value
        """
        # Warmup: не применяем топологический лосс в первые эпохи
        if epoch < warmup_epochs:
            return torch.tensor(0.0, device=logits.device)
        
        B, C, H, W = logits.shape
        
        # Получаем вероятности
        if self.from_logits:
            probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        else:
            probs = logits
        
        # Создаем one-hot encoding для targets
        targets_one_hot = F.one_hot(targets.long(), num_classes=C).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        
        # Вычисляем χ для предсказаний и ground truth для каждого класса
        chi_pred_list = []
        chi_gt_list = []
        
        for c in range(C):
            chi_pred_c = self._soft_euler_characteristic(probs[:, c, :, :])
            chi_gt_c = self._soft_euler_characteristic(targets_one_hot[:, c, :, :])
            chi_pred_list.append(chi_pred_c)
            chi_gt_list.append(chi_gt_c)
        
        # Stack: [C, B]
        chi_pred = torch.stack(chi_pred_list, dim=0)
        chi_gt = torch.stack(chi_gt_list, dim=0)
        
        # MSE loss для каждого класса
        loss_per_class = (chi_pred - chi_gt) ** 2  # [C, B]
        
        # Усредняем по батчу
        loss_per_class = loss_per_class.mean(dim=1)  # [C]
        
        # Применяем веса классов
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss = (loss_per_class * weights).sum() / weights.sum()
        else:
            loss = loss_per_class.mean()
        
        return loss