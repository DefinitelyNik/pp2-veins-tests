"""
Визуализация результатов обученной модели сегментации сосудов глазного дна.

Использование:
    python visualize_model.py                          # визуализация из валидационного набора
    python visualize_model.py --num-images 16          # количество изображений
    python visualize_model.py --custom-dir /path/imgs  # свои изображения
    python visualize_model.py --fold 1                 # номер фолда (веса)
"""

import argparse
import pathlib
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import albumentations as A

import segmentation_models_pytorch as smp
import common.settings_patches as settings
from common.data_preparation_patches import get_val_source_keys_per_fold


# ─── Настройки ──────────────────────────────────────────────────────────────────
WEIGHTS_PATH = pathlib.Path("UNETPP_WCD/weights_1_patches.pth")
OUTPUT_DIR = pathlib.Path("UNETPP_WCD/visualizations")

NUM_CLASSES = settings.NUM_CLASSES
CLASS_NAMES = settings.CLASS_NAMES
COLOR_MAP = settings.COLOR_MAP
PATCH_SIZE = settings.PATCH_SIZE
STRIDE = settings.STRIDE

ENCODER_NAME = "efficientnet-b0"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Цвета для визуализации
VIS_COLORS = {
    0: (0, 0, 0),          # background - чёрный
    1: (255, 0, 0),        # arteries - красный
    2: (0, 0, 255),        # veins - синий
    3: (255, 255, 0),      # crossings - жёлтый
    4: (0, 255, 0),        # capillaries - зелёный
}

TRANSFORM = A.Compose([
    A.Normalize(mean=MEAN, std=STD),
])


# ─── Загрузка модели ───────────────────────────────────────────────────────────
def load_model(weights_path: pathlib.Path, device: torch.device) -> smp.UnetPlusPlus:
    """Загружает обученную модель из чекпоинта."""
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
    ).to(device)

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✓ Модель загружена из {weights_path}")
    print(f"  Encoder: {ENCODER_NAME}, Classes: {NUM_CLASSES}")
    return model


# ─── Загрузка данных ───────────────────────────────────────────────────────────
def load_image_and_mask(
    img_path: pathlib.Path,
    mask_path: Optional[pathlib.Path] = None,
) -> tuple:
    """Загружает изображение и (опционально) маску."""
    image = np.array(Image.open(img_path).convert("RGB"))

    mask = None
    if mask_path is not None and mask_path.exists():
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
        for class_idx, color in COLOR_MAP.items():
            color_arr = np.array(color, dtype=np.uint8)
            mask[np.all(mask_rgb == color_arr, axis=-1)] = class_idx

    return image, mask


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Преобразует маску классов в цветное изображение."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in VIS_COLORS.items():
        colored[mask == class_idx] = color
    return colored


def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Создаёт наложение маски на изображение."""
    colored_mask = mask_to_color(mask)
    overlay = image.copy().astype(np.float32)
    overlay = overlay * (1 - alpha) + colored_mask.astype(np.float32) * alpha
    return overlay.astype(np.uint8)


# ─── Инференс на патчах ────────────────────────────────────────────────────────
def predict_full_image(
    model: smp.UnetPlusPlus,
    image: np.ndarray,
    device: torch.device,
    patch_size: int = 256,
    stride: int = 128,
) -> np.ndarray:
    """
    Предсказание на полном изображении через скользящее окно (патчи).
    Возвращает маску классов [H, W].
    """
    h, w = image.shape[:2]

    # Accumulators for logits
    logits_sum = np.zeros((NUM_CLASSES, h, w), dtype=np.float64)
    count_map = np.zeros((h, w), dtype=np.float64)

    # Pad image if smaller than patch_size
    pad_h = max(patch_size - h, 0)
    pad_w = max(patch_size - w, 0)
    if pad_h > 0 or pad_w > 0:
        image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        image_padded = image

    h_pad, w_pad = image_padded.shape[:2]

    # Sliding window
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            patch = image_padded[y:y + patch_size, x:x + patch_size]

            # Transform
            transformed = TRANSFORM(image=patch)["image"]
            tensor = torch.from_numpy(transformed).permute(2, 0, 1).float().unsqueeze(0)
            tensor = tensor.to(device)

            with torch.no_grad():
                logits = model(tensor)  # [1, C, PH, PW]
                probs = F.softmax(logits, dim=1)

            probs_np = probs.squeeze(0).cpu().numpy()  # [C, PH, PW]

            # Accumulate
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            py_end = y_end - y
            px_end = x_end - x

            logits_sum[:, y:y_end, x:x_end] += probs_np[:, :py_end, :px_end]
            count_map[y:y_end, x:x_end] += 1

    # Handle last row/column if not covered
    if count_map[-1, :].max() == 0:
        # Last row not covered, process it
        y = max(h - patch_size, 0)
        for x in range(0, w_pad - patch_size + 1, stride):
            patch = image_padded[y:y + patch_size, x:x + patch_size]
            transformed = TRANSFORM(image=patch)["image"]
            tensor = torch.from_numpy(transformed).permute(2, 0, 1).float().unsqueeze(0)
            tensor = tensor.to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
            probs_np = probs.squeeze(0).cpu().numpy()
            x_end = min(x + patch_size, w)
            px_end = x_end - x
            logits_sum[:, y:h, x:x_end] += probs_np[:, :h-y, :px_end]
            count_map[y:h, x:x_end] += 1

    # Average
    count_map = np.maximum(count_map, 1)
    logits_avg = logits_sum / count_map[np.newaxis, :, :]
    pred_mask = np.argmax(logits_avg, axis=0).astype(np.uint8)

    return pred_mask


# ─── Визуализация одного примера ──────────────────────────────────────────────
def visualize_single(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    title: str = "",
    save_path: Optional[pathlib.Path] = None,
    show: bool = False,
):
    """Визуализация: оригинал | предсказание | ground truth."""
    has_gt = gt_mask is not None
    ncols = 3 if has_gt else 2

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    # Prediction overlay
    pred_overlay = create_overlay(image, pred_mask, alpha=0.45)
    axes[1].imshow(pred_overlay)
    axes[1].set_title("Prediction", fontsize=12)
    axes[1].axis('off')

    # Ground truth overlay
    if has_gt:
        gt_overlay = create_overlay(image, gt_mask, alpha=0.45)
        axes[2].imshow(gt_overlay)
        axes[2].set_title("Ground Truth", fontsize=12)
        axes[2].axis('off')

    # Legend
    legend_patches = []
    for cls_idx, name in enumerate(CLASS_NAMES):
        color = np.array(VIS_COLORS.get(cls_idx, (128, 128, 128))) / 255.0
        legend_patches.append(mpatches.Patch(color=color, label=name))

    fig.legend(
        handles=legend_patches,
        loc='lower center',
        ncol=NUM_CLASSES,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  → Сохранено: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ─── Визуализация карты ошибок ────────────────────────────────────────────────
def visualize_error_map(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    save_path: Optional[pathlib.Path] = None,
):
    """Визуализация карты ошибок: где предсказание не совпадает с GT."""
    error_map = (pred_mask != gt_mask).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prediction
    axes[0].imshow(create_overlay(image, pred_mask, 0.45))
    axes[0].set_title("Prediction", fontsize=12)
    axes[0].axis('off')

    # Ground Truth
    axes[1].imshow(create_overlay(image, gt_mask, 0.45))
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis('off')

    # Error map
    axes[2].imshow(image)
    axes[2].imshow(error_map, cmap='Reds', alpha=0.5)
    axes[2].set_title("Error Map (red = mismatch)", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Per-class визуализация ────────────────────────────────────────────────────
def visualize_per_class(
    image: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: Optional[np.ndarray],
    save_path: Optional[pathlib.Path] = None,
):
    """Визуализация каждого класса отдельно (бинарные маски)."""
    ncols = NUM_CLASSES + 1  # +1 для оригинала
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')

    for c in range(NUM_CLASSES):
        pred_binary = (pred_mask == c).astype(np.float32)
        axes[c + 1].imshow(image)
        if pred_binary.sum() > 0:
            axes[c + 1].imshow(pred_binary, cmap='jet', alpha=0.5)
        axes[c + 1].set_title(f"Pred: {CLASS_NAMES[c]}", fontsize=9)
        axes[c + 1].axis('off')

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Основная логика ───────────────────────────────────────────────────────────
def get_val_images(num_images: int = 8):
    """Получает пути к валидационным изображениям."""
    val_keys_per_fold = get_val_source_keys_per_fold()
    # Берём первый фолд
    val_keys = val_keys_per_fold[0]

    images = []
    for key in val_keys[:num_images]:
        img_path = settings.NORMALIZED_IMG_DIR / f"{key}.png"
        msk_path = settings.NORMALIZED_MSK_DIR / f"{key}.png"
        if img_path.exists():
            images.append((img_path, msk_path))

    return images


def visualize_custom_dir(custom_dir: pathlib.Path, num_images: int = 8):
    """Получает пути к изображениям из пользовательской директории."""
    extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    img_paths = sorted([
        p for p in custom_dir.iterdir()
        if p.suffix.lower() in extensions
    ])[:num_images]

    return [(p, None) for p in img_paths]


def main():
    parser = argparse.ArgumentParser(description="Визуализация модели сегментации")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH),
                        help="Путь к весам модели")
    parser.add_argument("--num-images", type=int, default=8,
                        help="Количество изображений для визуализации")
    parser.add_argument("--custom-dir", type=str, default=None,
                        help="Директория со своими изображениями (без масок)")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Директория для сохранения результатов")
    parser.add_argument("--show", action="store_true",
                        help="Показывать изображения в окне (не только сохранять)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Загрузка модели
    model = load_model(pathlib.Path(args.weights), device)

    # Получение изображений
    if args.custom_dir:
        image_list = visualize_custom_dir(pathlib.Path(args.custom_dir), args.num_images)
        print(f"\nЗагружено {len(image_list)} изображений из {args.custom_dir}")
    else:
        image_list = get_val_images(args.num_images)
        print(f"\nЗагружено {len(image_list)} валидационных изображений")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Результаты будут сохранены в: {output_dir}\n")

    # Визуализация каждого изображения
    for idx, (img_path, msk_path) in enumerate(image_list):
        print(f"[{idx + 1}/{len(image_list)}] Обработка: {img_path.name}")

        # Загрузка
        image, gt_mask = load_image_and_mask(img_path, msk_path)

        # Предсказание
        pred_mask = predict_full_image(model, image, device,
                                       patch_size=PATCH_SIZE, stride=STRIDE)

        # 1. Основная визуализация
        vis_path = output_dir / f"{idx:03d}_{img_path.stem}_comparison.png"
        visualize_single(
            image, pred_mask, gt_mask,
            title=f"{img_path.stem}",
            save_path=vis_path,
            show=args.show,
        )

        # 2. Карта ошибок (если есть GT)
        if gt_mask is not None:
            err_path = output_dir / f"{idx:03d}_{img_path.stem}_errors.png"
            visualize_error_map(image, pred_mask, gt_mask, save_path=err_path)

        # 3. Per-class визуализация
        cls_path = output_dir / f"{idx:03d}_{img_path.stem}_per_class.png"
        visualize_per_class(image, pred_mask, gt_mask, save_path=cls_path)

    # Итоговая статистика
    print(f"\n{'='*60}")
    print(f"✓ Визуализация завершена!")
    print(f"  Сохранено в: {output_dir}")
    print(f"  Файлов: {len(list(output_dir.glob('*.png')))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()