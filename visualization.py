import matplotlib.pyplot as plt
import common.settings as settings
import common.data_preparation as data_prep

# --- ВЫБОР МОДЕЛИ (такой же, как в run_pipeline.py) ---
from IMPROVED_ARTEM_UNETPP.model import ImprovedUNetPlusPlus, train_extra_transforms, val_extra_transforms

ModelClass = ImprovedUNetPlusPlus

# --- ПАРАМЕТРЫ ВИЗУАЛИЗАЦИИ ---
NUM_SAMPLES = 5

def main():
    # Загружаем тестовые данные (без аугментаций)
    ((_, test_loader), (_, _), (_, _)) = data_prep.create_cross_val_loaders(
        val_extra_transforms=val_extra_transforms,
        batch_size=settings.BATCH_SIZE,
        num_workers=settings.NUM_WORKERS
    )

    # Инициализируем модель и загружаем сохранённые веса (если есть)
    model = ModelClass()
    model.load(ModelClass.get_model_save_path(1))   # предполагается статический метод
    print(f"Модель загружена из {ModelClass.get_model_save_path(1)}")

    # Собираем несколько примеров из test_loader
    vis_images = []
    vis_masks = []
    for batch_idx, (imgs, masks) in enumerate(test_loader):
        for i in range(len(imgs)):
            if len(vis_images) >= 100:
                break
            vis_images.append(imgs[i])
            vis_masks.append(masks[i])
        if len(vis_images) >= 100:
            break


    model.visualize_sample(vis_images[10], vis_masks[10], None, None, None)
    # Создаём сетку подграфиков
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(15, 5 * NUM_SAMPLES))
    for i in range(NUM_SAMPLES):
        ax_img = axes[i, 0]
        ax_truth = axes[i, 1]
        ax_pred = axes[i, 2]
        model.visualize_sample(vis_images[i], vis_masks[i], ax_img, ax_truth, ax_pred)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
