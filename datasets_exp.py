import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------- НАСТРОЙКИ ----------
SRC_DIR = Path("dataset")
DST_DIR = Path("normalized_dataset")
IMG_SUBDIR = "images"
MSK_SUBDIR = "masks"

FOV_THRESHOLD = 10
MARGIN = 5
PAD_MODE = "constant"
TARGET_SIZE = (1280, 1280)

NUM_CLASSES = 5

# BGR-цвета (порядок как в cv2.imread)
COLOR_TO_CLASS = {
    (0,   0,   0):   0,   # фон
    (255, 0,   0):   1,   # артерии
    (0,   0,   255): 2,   # вены
    (0,   255, 0):   3,   # пересечения
    (255, 255, 255): 4,   # неопределённые
}

# Обратная карта: индекс -> BGR-цвет
INDEX_TO_COLOR = {v: k for k, v in COLOR_TO_CLASS.items()}

UNKNOWN_AS = 0
FOV_DETECT_SCALE = 0.25
NUM_WORKERS = None
PRINT_CLASS_STATS = True

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
# --------------------------------


def find_fov_bbox_fast(img_bgr, threshold, margin, scale):
    H, W = img_bgr.shape[:2]
    if scale < 1.0:
        small = cv2.resize(img_bgr, (max(1, int(W * scale)), max(1, int(H * scale))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr
        scale = 1.0

    gray = small.max(axis=2)
    mask = (gray > threshold).astype(np.uint8) * 255
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return 0, 0, W, H

    areas = stats[1:, cv2.CC_STAT_AREA]
    best = 1 + int(np.argmax(areas))

    inv = 1.0 / scale
    x = int(stats[best, cv2.CC_STAT_LEFT] * inv)
    y = int(stats[best, cv2.CC_STAT_TOP] * inv)
    w = int(stats[best, cv2.CC_STAT_WIDTH] * inv)
    h = int(stats[best, cv2.CC_STAT_HEIGHT] * inv)

    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(W - x, w + 2 * margin)
    h = min(H - y, h + 2 * margin)
    return x, y, w, h


def crop_and_square(img, bbox, pad_mode):
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w]
    ch, cw = crop.shape[:2]
    side = max(ch, cw)

    pad_top = (side - ch) // 2
    pad_bottom = side - ch - pad_top
    pad_left = (side - cw) // 2
    pad_right = side - cw - pad_left

    border = cv2.BORDER_REFLECT_101 if pad_mode == "reflect" else cv2.BORDER_CONSTANT

    if crop.ndim == 2:
        return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                  border, value=0)
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                              border, value=(0, 0, 0))


def resize_image(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def color_mask_to_index(mask_bgr, color_to_class, unknown_as=0, num_classes=None):
    """Цветная BGR-маска -> индексная (H, W) с метками классов."""
    H, W = mask_bgr.shape[:2]
    out = np.full((H, W), unknown_as, dtype=np.uint8)

    for color, cls in color_to_class.items():
        m = np.all(mask_bgr == np.array(color, dtype=mask_bgr.dtype), axis=2)
        out[m] = cls

    if num_classes is not None and out.max() >= num_classes:
        raise ValueError(f"Класс {out.max()} превышает NUM_CLASSES={num_classes}")
    return out


def resize_multiclass_mask(mask, size, num_classes):
    """Индексная маска -> ресайз через по-канальный one-hot -> argmax."""
    out_w, out_h = size
    acc = np.zeros((num_classes, out_h, out_w), dtype=np.float32)
    for c in range(num_classes):
        ch = (mask == c).astype(np.uint8) * 255
        acc[c] = cv2.resize(ch, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    return np.argmax(acc, axis=0).astype(mask.dtype)


def index_to_color_mask(mask_idx, index_to_color):
    """Индексная маска (H, W) -> цветная BGR (H, W, 3) по заданным цветам."""
    H, W = mask_idx.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in index_to_color.items():
        out[mask_idx == cls] = color
    return out


def process_one(args):
    (img_path_str, msk_path_str, out_img_path_str, out_msk_path_str,
     target_size, num_classes, fov_threshold, margin, pad_mode, fov_scale,
     print_stats, color_to_class, unknown_as, index_to_color) = args

    img_path = Path(img_path_str)
    msk_path = Path(msk_path_str)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
    if img is None or msk is None:
        return ("skip", img_path.name, "read error")

    # Цветная маска -> индексная
    if msk.ndim == 3:
        msk_idx = color_mask_to_index(msk, color_to_class, unknown_as=unknown_as,
                                      num_classes=num_classes)
    else:
        msk_idx = msk

    bbox = find_fov_bbox_fast(img, fov_threshold, margin, fov_scale)

    img_crop = crop_and_square(img, bbox, pad_mode)
    msk_crop = crop_and_square(msk_idx, bbox, pad_mode)

    # Ресайз изображения
    if img_crop.shape[:2] != (target_size[1], target_size[0]):
        img_out = resize_image(img_crop, target_size)
    else:
        img_out = img_crop

    # Ресайз индексной маски
    if msk_crop.shape[:2] != (target_size[1], target_size[0]):
        msk_idx_out = resize_multiclass_mask(msk_crop, target_size, num_classes)
    else:
        msk_idx_out = msk_crop

    # Индексная -> цветная BGR
    msk_out = index_to_color_mask(msk_idx_out, index_to_color)

    # Сохранение
    cv2.imwrite(out_img_path_str, img_out)
    cv2.imwrite(out_msk_path_str, msk_out)

    info = None
    if print_stats:
        before = np.bincount(msk_crop.ravel(), minlength=num_classes)[:num_classes]
        after = np.bincount(msk_idx_out.ravel(), minlength=num_classes)[:num_classes]
        info = (img_path.name, before.tolist(), after.tolist())

    return ("ok", img_path.name, info)


def main():
    img_dir = SRC_DIR / IMG_SUBDIR
    msk_dir = SRC_DIR / MSK_SUBDIR

    assert img_dir.is_dir(), f"Нет папки {img_dir}"
    assert msk_dir.is_dir(), f"Нет папки {msk_dir}"

    out_img_dir = DST_DIR / IMG_SUBDIR
    out_msk_dir = DST_DIR / MSK_SUBDIR
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Найдено изображений: {len(img_files)}")

    tasks = []
    skipped = []
    for img_path in img_files:
        msk_path = None
        for ext in IMG_EXTS:
            cand = msk_dir / (img_path.stem + ext)
            if cand.exists():
                msk_path = cand
                break
        if msk_path is None:
            skipped.append(img_path.name)
            continue

        out_img_path = out_img_dir / img_path.name
        out_msk_path = out_msk_dir / (img_path.stem + msk_path.suffix)

        tasks.append((
            str(img_path), str(msk_path),
            str(out_img_path), str(out_msk_path),
            TARGET_SIZE, NUM_CLASSES,
            FOV_THRESHOLD, MARGIN, PAD_MODE, FOV_DETECT_SCALE,
            PRINT_CLASS_STATS,
            COLOR_TO_CLASS, UNKNOWN_AS, INDEX_TO_COLOR,
        ))

    workers = NUM_WORKERS or os.cpu_count() or 1
    print(f"Воркеров: {workers}")

    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(process_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Обработка"):
            results.append(f.result())

    n_ok = 0
    stats_rows = []
    for status, name, info in results:
        if status == "ok":
            n_ok += 1
            if info is not None:
                stats_rows.append(info)
        else:
            skipped.append(name)

    if skipped:
        print(f"\nПропущено: {len(skipped)}")
        for s in skipped[:10]:
            print("  ", s)

    if PRINT_CLASS_STATS and stats_rows:
        before_sum = np.zeros(NUM_CLASSES, dtype=np.int64)
        after_sum = np.zeros(NUM_CLASSES, dtype=np.int64)
        for _, b, a in stats_rows:
            before_sum += np.array(b, dtype=np.int64)
            after_sum += np.array(a, dtype=np.int64)

        print("\nСтатистика по классам (суммарно по всем изображениям):")
        print(f"{'class':>6} {'before':>14} {'after':>14} {'keep %':>8}")
        for c in range(NUM_CLASSES):
            b = before_sum[c]
            a = after_sum[c]
            pct = (a / b * 100) if b > 0 else float("nan")
            print(f"{c:>6} {b:>14} {a:>14} {pct:>7.1f}%")

    print(f"\nГотово. Успешно: {n_ok}/{len(tasks)}")
    print(f"Результат в: {DST_DIR.resolve()}")


if __name__ == "__main__":
    main()