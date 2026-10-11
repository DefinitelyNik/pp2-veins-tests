import numpy as np
import pandas as pd
from pathlib import Path

# ---------- настройки ----------
FILES = [
    "DL-CE.csv",
    "DL-DF.csv",
    "DL-T.csv",
    "DL-WCD.csv",
    "TFFM-CE.csv",
    "UN-DF.csv",
    "UN-WCD.csv",
    "ASSEMBLE.csv",
]

CLASSES = ["background", "arteries", "veins", "crossings", "capillaries"]

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("metrics_per_class")


def per_class_dice_iou(precision: pd.Series, recall: pd.Series):
    """Векторизованный расчёт Dice и IoU из precision/recall (Series)."""
    precision = precision.astype(float)
    recall = recall.astype(float)

    denom_dice = precision + recall
    denom_iou = precision + recall - precision * recall

    # np.where корректно работает с Series и не падает на нулях
    dice = np.where(denom_dice != 0, 2 * precision * recall / denom_dice, 0.0)
    iou = np.where(denom_iou != 0, precision * recall / denom_iou, 0.0)

    return pd.Series(dice, index=precision.index), pd.Series(iou, index=precision.index)


def process_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()

    for cls in CLASSES:
        p = out[f"{cls}_precision"]
        r = out[f"{cls}_recall"]
        dice_c, iou_c = per_class_dice_iou(p, r)
        out[f"{cls}_dice"] = dice_c
        out[f"{cls}_iou"] = iou_c

    # убираем старые общие dice/iou
    out = out.drop(columns=[c for c in ["dice", "iou"] if c in out.columns])

    # порядок колонок: для каждого класса — accuracy, precision, recall, dice, iou
    ordered_cols = []
    for cls in CLASSES:
        for metric in ["accuracy", "precision", "recall", "dice", "iou"]:
            col = f"{cls}_{metric}"
            if col in out.columns:
                ordered_cols.append(col)

    rest = [c for c in out.columns if c not in ordered_cols]
    return out[ordered_cols + rest]


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    for fname in FILES:
        src = INPUT_DIR / fname
        if not src.exists():
            print(f"[skip] {fname} не найден")
            continue

        new_df = process_file(src)
        dst = OUTPUT_DIR / fname
        new_df.to_csv(dst, index=False)
        print(f"[ok] {dst}  shape={new_df.shape}")


if __name__ == "__main__":
    main()