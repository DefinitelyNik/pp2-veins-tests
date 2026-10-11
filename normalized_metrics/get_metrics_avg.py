import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- настройки ----------
METRICS_DIR = Path("metrics_per_class")
OUTPUT_CSV  = Path("weighted_metrics_mean_std.csv")

CLASSES = ["veins", "arteries", "crossings"]

WEIGHTS = {
    "veins":     0.5366725783844343,
    "arteries":  0.44476505347681267,
    "crossings": 0.018562368138753074,
}

METRICS = ["dice", "iou", "precision", "recall", "accuracy"]


def load_all_metrics(metrics_dir: Path) -> pd.DataFrame:
    frames = []
    for fname in sorted(os.listdir(metrics_dir)):
        if not fname.lower().endswith(".csv"):
            continue
        path = metrics_dir / fname
        df = pd.read_csv(path)
        df["model"] = Path(fname).stem
        df["fold"] = range(len(df))
        frames.append(df)

    if not frames:
        raise RuntimeError(f"В папке {metrics_dir} не найдено CSV-файлов")

    return pd.concat(frames, ignore_index=True)


def weighted_metric_for_row(row: pd.Series, weights: dict, metric: str) -> float:
    """Weighted-метрика для одной строки (один фолд одной модели)."""
    total_w = sum(weights.values())
    value = 0.0
    for cls, w in weights.items():
        value += (w / total_w) * row[f"{cls}_{metric}"]
    return value


def fmt_mean_std(mean: float, std: float, decimals: int = 3) -> str:
    """Форматирует 'MEAN ± STD' с запятой как десятичным разделителем."""
    m = f"{mean:.{decimals}f}".replace(".", ",")
    s = f"{std:.{decimals}f}".replace(".", ",")
    return f"{m} ± {s}"


def main():
    df = load_all_metrics(METRICS_DIR)

    missing = [
        f"{cls}_{metric}"
        for cls in CLASSES
        for metric in METRICS
        if f"{cls}_{metric}" not in df.columns
    ]
    if missing:
        raise KeyError(f"В файлах не хватает колонок: {missing}")

    # 1) weighted-метрики внутри каждого фолда
    fold_rows = []
    for _, row in df.iterrows():
        rec = {"model": row["model"], "fold": row["fold"]}
        for metric in METRICS:
            rec[metric] = weighted_metric_for_row(row, WEIGHTS, metric)
        fold_rows.append(rec)

    fold_df = pd.DataFrame(fold_rows)

    # 2) MEAN ± STD по фолдам для каждой модели
    summary_rows = []
    for model, sub in fold_df.groupby("model"):
        rec = {"model": model}
        for metric in METRICS:
            mean = sub[metric].mean()
            std  = sub[metric].std(ddof=1) if len(sub) > 1 else 0.0
            rec[metric] = fmt_mean_std(mean, std)
        summary_rows.append(rec)

    result = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)

    print("Weighted-метрики (MEAN ± STD по фолдам), классы: veins, arteries, crossings")
    print(f"Веса: {WEIGHTS}")
    print()
    print(result.to_string(index=False))

    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nСохранено: {OUTPUT_CSV}")

    # бонус: сохранить и «сырые» значения по фолдам — пригодится для проверки
    fold_df.to_csv("weighted_metrics_per_fold.csv", index=False)
    print("Сохранено: weighted_metrics_per_fold.csv")


if __name__ == "__main__":
    main()