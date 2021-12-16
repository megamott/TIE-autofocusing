import numpy as np
from PIL import Image
from generation.normalizer import normalize


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

    img = Image.open(path)
    gray_mode = img.mode

    if gray_mode == gray_8bit:
        old_max = 2 ** 8 - 1  # 255

    elif gray_mode == gray_16bit:
        old_max = 2 ** 16 - 1  # 65 535

    else:  # color-image
        img = img.convert(gray_8bit)
        old_max = 2 ** 8 - 1

    return normalize(np.asarray(img, np.float64), old_min=0, old_max=old_max)
