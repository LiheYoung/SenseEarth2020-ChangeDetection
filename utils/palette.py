import numpy as np
import os
from PIL import Image
from tqdm import tqdm


def color_map():
    cmap = np.zeros((7, 3), dtype=np.uint8)
    cmap[0] = np.array([255, 255, 255])
    cmap[1] = np.array([0, 0, 255])
    cmap[2] = np.array([128, 128, 128])
    cmap[3] = np.array([0, 128, 0])
    cmap[4] = np.array([0, 255, 0])
    cmap[5] = np.array([128, 0, 0])
    cmap[6] = np.array([255, 0, 0])

    return cmap


if __name__ == '__main__':
    path = '/data/lihe/datasets/SenseEarth/ChangeDetection/train/label2'
    filenames = os.listdir(path)

    cmap = color_map()

    for filename in tqdm(filenames):
        mask = Image.open(os.path.join(path, filename)).convert("P")
        mask.putpalette(cmap)
        mask.save(os.path.join(path, filename))
