import os
import numpy as np

from typing import List
from PIL import Image
from miscellaneous.normalizer import normalize


NDARRAY_EXTENSION = '.npy'


def load_files(paths: List[str]) -> List[np.ndarray]:
    return [load_file(path) for path in paths]


def load_file(path: str) -> np.ndarray:
    _, extension = os.path.splitext(path)

    if extension == NDARRAY_EXTENSION:
        array = np.load(path)
    else:
        array = load_image(path)

    return array


def load_image(path: str) -> np.ndarray:
    """
    Загружает изображение, конвертирует его в numpy.ndarray (dtype=np.float64), приводит к динамическому диапазону
    [0.0 ... 1.0].
    Цветные изображения конвертируются в полутоновые.
    :param path: путь к файлу
    :return матрица
    """
    gray_8bit = 'L'
    gray_16bit = 'I;16'
    max_8bit = 255
    max_16bit = 65_535

    img = Image.open(path)
    gray_mode = img.mode

    if gray_mode == gray_8bit:
        old_max = max_8bit

    elif gray_mode == gray_16bit:
        old_max = max_16bit

    else:  # color-image
        img = img.convert(gray_8bit)
        old_max = max_8bit

    return normalize(np.asarray(img, np.float64), old_min=0, old_max=old_max)
