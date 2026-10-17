from collections import Counter
import os
import numpy as np
from PIL import Image


IMAGES_DIR = 'images/'
MASKS_DIR = 'masks/'
images_files = [f'{IMAGES_DIR}{filename}' for filename in os.listdir(IMAGES_DIR)]
masks_files = [f'{MASKS_DIR}{filename}' for filename in os.listdir(MASKS_DIR)]

resolutions = dict()
for file in images_files:
    resolutions[file] = Image.open(file).size

print(resolutions)
print(set(resolutions.values()))


colors_counts = Counter()
for i, file in enumerate(masks_files):
    print(f'{i + 1} / {len(masks_files)}')
    img = Image.open(file).convert('RGB')
    colors_counts.update(img.getdata())


colors_counts = dict(colors_counts)

VEINS = (0, 0, 255)
ARTERIES = (255, 0, 0)
CROSSINGS = (0, 255, 0)

total_count = sum([colors_counts[class_] for class_ in [VEINS, ARTERIES, CROSSINGS]])
weights = {
    class_: colors_counts[class_] / total_count
    for class_ in [VEINS, ARTERIES, CROSSINGS]
}

print(weights)
