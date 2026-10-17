# pp2-veins

Сейчас мы занимаемся тем, что делаем гипотезы с наший топовой моделью - UNETPP_WCD - U-Net++ с комбинированной функцией потерь: Dice + взвешенная кросс-энтропия

## Патчи БЕЗ перекрытия

Для того, чтобы поиграться с моделью, которая обучается на патчах без перекрытия, нам нужно менять только файлы с *_patches.py.
Чтобы прогнать на внешних датасетах, запускаем run_for_new_images_patches.py

## Патчи С перекрытием

Для того, чтобы поиграться с моделью, которая обучается на патчах с перекрытием, нам нужно менять только файлы с *_patches_overlap.py.
Чтобы прогнать на внешних датасетах, запускаем run_for_new_images_patches_overlap.py

Табличкой можно показать так:

| Файл | БЕЗ перекрытия | С перекрытием |
|:---:|:---:|:---:|
| Датасет | dataset_patches_256 | dataset_patches_256_overlap |
| Настройки | settings_patches.py | settings_patches_overlap.py |
| Подготовка данных | data_preparation_patches.py | data_preparation_patches_overlap.py |
| Метрики | metrics_patches.py | metrics_patches_overlap.py |
| Модель | model_patches.py | model_patches_overlap.py |
| Пайплайн | run_pipeline_UNET_WCD_patches.py | run_pipeline_UNET_WCD_patches_overlap.py |
| Прогон на внешнем датасете | run_for_new_images_patches.py | run_for_new_images_patches_overlap.py |

