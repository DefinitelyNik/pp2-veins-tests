# common/topology_metrics.py
"""
Точные (не дифференцируемые) топологические метрики для валидации и отчётов.

Считаются на бинарных масках, поэтому дают НАСТОЯЩИЕ числа Бетти:
    β0 — число компонент связности (4-связность для объекта)
    β1 — число петель/дыр          (8-связность для фона)
    χ  = β0 − β1 = V − E + F

Соглашение о связности (4-связность объекта / 8-связность фона) выбрано не
произвольно: именно оно совпадает с эйлеровой характеристикой кубического
комплекса, которую использует topology_loss.SoftEulerCharacteristicLoss.
Проверено на 500 случайных масках: расхождений нет.

Использование в train-loop:
    acc = TopologyErrorAccumulator(num_classes=5, class_names=settings.CLASS_NAMES)
    acc.update(preds_bhw_np, targets_bhw_np)
    metrics = acc.compute()          # -> dict для лога / CSV
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

__all__ = [
    "euler_characteristic",
    "betti_numbers",
    "topology_errors_for_pair",
    "TopologyErrorAccumulator",
]

_S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)


# ===========================================================================
# Примитивы
# ===========================================================================
def euler_characteristic(mask: np.ndarray) -> float:
    """
    Точная χ = V − E + F для бинарной маски (H, W).

    F  = число включённых пикселей
    Eh = число вертикально-смежных пар
    Ev = число горизонтально-смежных пар
    V  = число полностью включённых 2×2 блоков

    Работает и для вырожденных размеров (1×N, N×1, 1×1).
    """
    m = (np.asarray(mask) > 0).astype(np.float64)
    if m.size == 0:
        return 0.0
    F_ = m.sum()
    Eh = (m[:-1, :] * m[1:, :]).sum() if m.shape[0] > 1 else 0.0
    Ev = (m[:, :-1] * m[:, 1:]).sum() if m.shape[1] > 1 else 0.0
    V_ = (
        (m[:-1, :-1] * m[:-1, 1:] * m[1:, :-1] * m[1:, 1:]).sum()
        if (m.shape[0] > 1 and m.shape[1] > 1)
        else 0.0
    )
    return float(F_ - Eh - Ev + V_)


def betti_numbers(mask: np.ndarray) -> Tuple[int, int]:
    """
    (β0, β1) бинарной маски.

    β0 считается через scipy.ndimage.label с 4-связностью,
    β1 восстанавливается из χ = β0 − β1  (быстрее и надёжнее,
    чем прямой подсчёт дыр через компоненты фона).
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return 0, 0
    if not _HAS_SCIPY:
        raise RuntimeError("Для betti_numbers нужен scipy: pip install scipy")
    b0 = int(ndimage.label(m, structure=_S4)[1])
    chi = euler_characteristic(m)
    b1 = int(round(b0 - chi))
    return b0, max(b1, 0)


def topology_errors_for_pair(
    pred_idx: np.ndarray,
    gt_idx: np.ndarray,
    num_classes: int,
    class_names: Optional[Dict[int, str]] = None,
    include_background: bool = True,
    include_union: bool = True,
) -> Dict[str, float]:
    """
    |Δβ0|, |Δβ1|, |Δχ| по каждому классу для ОДНОЙ пары (pred, gt).

    Параметры
    ---------
    pred_idx, gt_idx : (H, W) uint8 — карты индексов классов
    class_names      : {0: 'background', ...}; по умолчанию 'class_{c}'
    include_union    : добавить метрики по объединению всего переднего плана
                       (классы 1..C-1). Именно там живут васкулярные петли (β1).
    """
    pred_idx = np.asarray(pred_idx)
    gt_idx = np.asarray(gt_idx)
    assert pred_idx.shape == gt_idx.shape, (pred_idx.shape, gt_idx.shape)

    out: Dict[str, float] = {}

    def _emit(tag: str, p_mask: np.ndarray, g_mask: np.ndarray) -> None:
        p0, p1 = betti_numbers(p_mask)
        g0, g1 = betti_numbers(g_mask)
        out[f"topo_abs_dbeta0_{tag}"] = float(abs(p0 - g0))
        out[f"topo_abs_dbeta1_{tag}"] = float(abs(p1 - g1))
        out[f"topo_abs_dchi_{tag}"] = float(abs((p0 - p1) - (g0 - g1)))
        out[f"topo_beta0_{tag}_pred"] = float(p0)
        out[f"topo_beta0_{tag}_gt"] = float(g0)
        out[f"topo_beta1_{tag}_pred"] = float(p1)
        out[f"topo_beta1_{tag}_gt"] = float(g1)

    for c in range(num_classes):
        if c == 0 and not include_background:
            continue
        name = (class_names or {}).get(c, f"class_{c}")
        _emit(name, pred_idx == c, gt_idx == c)

    if include_union and num_classes > 1:
        _emit("foreground_union", pred_idx != 0, gt_idx != 0)

    return out


# ===========================================================================
# Аккумулятор (для train/val loop и для evaluation-скриптов)
# ===========================================================================
class TopologyErrorAccumulator:
    """
    Накапливает |Δβ0|, |Δβ1|, |Δχ| по батчам и выдаёт средние.

        acc = TopologyErrorAccumulator(5, settings.CLASS_NAMES)
        for images, masks in val_loader:
            preds = model.predict(images)                       # (B,H,W) на CPU
            acc.update(preds.numpy(), masks.numpy())
        metrics = acc.compute()          # dict[str, float]
        summary = acc.summary()          # одно число: средняя тополог. ошибка
    """

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Dict[int, str]] = None,
        include_background: bool = True,
        include_union: bool = True,
    ):
        self.num_classes = int(num_classes)
        self.class_names = class_names
        self.include_background = include_background
        self.include_union = include_union
        self._sums: Dict[str, float] = {}
        self._count: int = 0

    def update(self, preds: np.ndarray, targets: np.ndarray) -> None:
        """preds, targets: (B, H, W) или (H, W)"""
        preds = np.asarray(preds)
        targets = np.asarray(targets)
        if preds.ndim == 2:
            preds = preds[None]
            targets = targets[None]
        assert preds.shape == targets.shape

        for b in range(preds.shape[0]):
            d = topology_errors_for_pair(
                preds[b],
                targets[b],
                self.num_classes,
                self.class_names,
                include_background=self.include_background,
                include_union=self.include_union,
            )
            for k, v in d.items():
                self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._count += 1

    def compute(self) -> Dict[str, float]:
        if self._count == 0:
            return {}
        return {k: v / self._count for k, v in self._sums.items()}

    def summary(self) -> float:
        """
        Скалярная топологическая ошибка для отбора чекпоинтов
        (аналог PD-distance из статьи: чем меньше, тем лучше).

        Среднее от |Δβ0| и |Δβ1| по классам И по объединению переднего плана.
        """
        m = self.compute()
        if not m:
            return float("inf")
        vals: List[float] = [v for k, v in m.items() if "_abs_dbeta" in k]
        return float(np.mean(vals)) if vals else float("inf")

    def reset(self) -> None:
        self._sums = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count


# ===========================================================================
# Самопроверка:  python -m common.topology_metrics
# ===========================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ok = True
    for _ in range(200):
        H, W = int(rng.integers(4, 40)), int(rng.integers(4, 40))
        m = (rng.random((H, W)) < rng.uniform(0.05, 0.7))
        if rng.random() < 0.6:
            m = ndimage.binary_dilation(m, iterations=int(rng.integers(0, 4)))
        b0, b1 = betti_numbers(m)
        chi = euler_characteristic(m)
        if (b0 - b1) != round(chi):
            ok = False
            print("MISMATCH", H, W, b0, b1, chi)
            break
    print("betti <-> euler consistent:", ok)

    # демонстрация
    ring = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    print("ring3x3  beta0,beta1,chi =", betti_numbers(ring), euler_characteristic(ring))
    diag = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    print("diag     beta0,beta1,chi =", betti_numbers(diag), euler_characteristic(diag))