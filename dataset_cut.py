import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------- НАСТРОЙКИ ----------
SRC_DIR = Path("normalized_dataset")
IMG_SUBDIR = "images"
MSK_SUBDIR = "masks"

DST_NO_OVERLAP = Path("dataset_patches_256")
DST_OVERLAP = Path("dataset_patches_256_overlap")

PATCH_SIZE = 256
STRIDE_NO_OVERLAP = 256
STRIDE_OVERLAP = 128

# Фильтрация: если доля "буквально чёрных" (0,0,0) пикселей БОЛЬШЕ этого —
# патч отбрасывается.
MAX_BLACK_RATIO = 0.5

NUM_CLASSES = 5

# BGR-цвета (те же, что в normalized_dataset)
COLOR_TO_CLASS = {
    (0,   0,   0):   0,   # фон
    (255, 0,   0):   1,   # артерии
    (0,   0,   255): 2,   # вены
    (0,   255, 0):   3,   # пересечения
    (255, 255, 255): 4,   # неопределённые
}
UNKNOWN_AS = 0

NUM_WORKERS = None

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
# --------------------------------


def color_mask_to_index(mask_bgr, color_to_class, unknown_as=0):
    """Цветная BGR-маска -> индексная (H, W)."""
    H, W = mask_bgr.shape[:2]
    out = np.full((H, W), unknown_as, dtype=np.uint8)
    for color, cls in color_to_class.items():
        m = np.all(mask_bgr == np.array(color, dtype=mask_bgr.dtype), axis=2)
        out[m] = cls
    return out


def iter_patch_coords(H, W, patch, stride):
    ys = list(range(0, H - patch + 1, stride))
    xs = list(range(0, W - patch + 1, stride))
    if ys[-1] + patch < H:
        ys.append(H - patch)
    if xs[-1] + patch < W:
        xs.append(W - patch)
    for y in ys:
        for x in xs:
            yield y, x


def compute_black_mask(img):
    """True там, где пиксель строго чёрный (0,0,0) / для grayscale == 0."""
    if img.ndim == 3:
        return (img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0)
    else:
        return (img == 0)


def save_patches_for_one(args):
    (img_path_str, msk_path_str, out_img_dir_str, out_msk_dir_str,
     patch_size, stride, max_black_ratio,
     color_to_class, unknown_as) = args

    img_path = Path(img_path_str)
    msk_path = Path(msk_path_str)

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    msk = cv2.imread(str(msk_path), cv2.IMREAD_UNCHANGED)
    if img is None or msk is None:
        return ("skip", img_path.name, "read error", 0, 0)

    # Приводим маску к цветному BGR (если вдруг пришла индексная — раскрасим).
    # В normalized_dataset маски уже цветные BGR, но подстрахуемся.
    if msk.ndim == 2:
        # индексная -> цветная
        INDEX_TO_COLOR = {v: k for k, v in color_to_class.items()}
        msk_bgr = np.zeros((*msk.shape, 3), dtype=np.uint8)
        for cls, color in INDEX_TO_COLOR.items():
            msk_bgr[msk == cls] = color
    else:
        msk_bgr = msk

    H, W = img.shape[:2]
    if H < patch_size or W < patch_size:
        return ("skip", img_path.name, f"too small {H}x{W}", 0, 0)

    out_img_dir = Path(out_img_dir_str)
    out_msk_dir = Path(out_msk_dir_str)
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    stem = img_path.stem
    saved = 0
    skipped = 0

    black_full = compute_black_mask(img)

    for y, x in iter_patch_coords(H, W, patch_size, stride):
        black_patch = black_full[y:y + patch_size, x:x + patch_size]
        black_ratio = float(black_patch.mean())
        if black_ratio > max_black_ratio:
            skipped += 1
            continue

        i_patch = img[y:y + patch_size, x:x + patch_size]
        m_patch = msk_bgr[y:y + patch_size, x:x + patch_size]

        name = f"{stem}_y{y}_x{x}.png"
        cv2.imwrite(str(out_img_dir / name), i_patch)
        cv2.imwrite(str(out_msk_dir / name), m_patch)
        saved += 1

    return ("ok", img_path.name, "", saved, skipped)


def run_mode(src_dir, dst_dir, stride, mode_name):
    img_dir = src_dir / IMG_SUBDIR
    msk_dir = src_dir / MSK_SUBDIR

    assert img_dir.is_dir(), f"Нет папки {img_dir}"
    assert msk_dir.is_dir(), f"Нет папки {msk_dir}"

    out_img_dir = dst_dir / IMG_SUBDIR
    out_msk_dir = dst_dir / MSK_SUBDIR
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"\n[{mode_name}] Найдено изображений: {len(img_files)}")
    print(f"[{mode_name}] patch={PATCH_SIZE}, stride={stride}, "
          f"max_black_ratio={MAX_BLACK_RATIO}")

    tasks = []
    for img_path in img_files:
        msk_path = None
        for ext in IMG_EXTS:
            cand = msk_dir / (img_path.stem + ext)
            if cand.exists():
                msk_path = cand
                break
        if msk_path is None:
            continue

        tasks.append((
            str(img_path), str(msk_path),
            str(out_img_dir), str(out_msk_dir),
            PATCH_SIZE, stride, MAX_BLACK_RATIO,
            COLOR_TO_CLASS, UNKNOWN_AS,
        ))

    workers = NUM_WORKERS or os.cpu_count() or 1
    print(f"[{mode_name}] Воркеров: {workers}")

    total_saved = 0
    total_skipped = 0
    failed = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(save_patches_for_one, t) for t in tasks]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc=f"[{mode_name}]"):
            status, name, msg, saved, skipped = f.result()
            if status == "ok":
                total_saved += saved
                total_skipped += skipped
            else:
                failed.append((name, msg))

    if failed:
        print(f"[{mode_name}] Проблемных файлов: {len(failed)}")
        for n, m in failed[:5]:
            print("  ", n, m)

    print(f"[{mode_name}] Сохранено патчей: {total_saved}, "
          f"отфильтровано: {total_skipped}")
    print(f"[{mode_name}] Результат в: {dst_dir.resolve()}")


def main():
    run_mode(SRC_DIR, DST_NO_OVERLAP, STRIDE_NO_OVERLAP, "no_overlap")
    run_mode(SRC_DIR, DST_OVERLAP, STRIDE_OVERLAP, "overlap")


if __name__ == "__main__":
    main()