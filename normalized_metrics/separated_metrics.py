import os

import numpy as np
import pandas as pd


def format_mean_std(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # выборочное стандартное отклонение
    # Округляем до 3 знаков, заменяем точку на запятую
    return f"{mean:.3f} ± {std:.3f}".replace('.', ',')


metric_files = [
    filename
    for filename in os.listdir()
    if (
        filename.endswith('.csv')
        and 'ASSEMBLE' not in filename
        and 'real_metrics' not in filename
    )]
metric_columns = [
    f'{class_}_{metric_name}'
    for metric_name in ['dice', 'iou', 'precision', 'recall']
    for class_ in ['veins', 'arteries', 'crossings']
]

metrics = [
    (metric_file.split('.')[0], pd.read_csv(metric_file))
    for metric_file in metric_files
]

columns = ['model'] + metric_columns
total = pd.DataFrame(columns=columns)

for metric in metrics:
    total.loc[len(total)] = {
        'model': metric[0]
    } | {
        col: format_mean_std(metric[1][col])
        for col in metric_columns
    }
total.to_excel('Classes precision recall dice iou.xlsx')

# print(format_mean_std(metrics['DL-CE']['veins_recall']))
