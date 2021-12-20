from typing import Union
import numpy as np


def rect_1d(x, a=1., w=1., x0=0., y0=0):
    """
    Возвращает 1-мерную прямоугольную функцию
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ширина
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :return: np.ndarray
    """
    return a * (np.abs((x - x0) / w) < 0.5) + y0


def rect_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0., z0=0):
    """
    Возвращает 2-мерную прямоугольную функцию
    :param x: np.ndarray 2-мерная координатная сетка по оси X
    :param y: np.ndarray 2-мерная координатная сетка по оси Y
    :param a: Union[float, int] амплитуда
    :param wx: Union[float, int] ширина по оси X
    :param wy: Union[float, int] ширина по оси Y
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :param z0: Union[float, int] смещение относительно нуля координат по оси Z
    :return: np.ndarray
    """
    return a * (rect_1d(x, w=wx, x0=x0) * rect_1d(y, w=wy, x0=y0)) + z0


def circ(r, a=1., w=1., r0=0.):
    return a * rect_1d(r, w=w, x0=r0)


def circ_cartesian(x, y, a=1., w=1., x0=0., y0=0.):
    return a * ((np.sqrt((x - x0) ** 2 + (y - y0) ** 2) / w) < 0.5)


def triangle_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную треугольную функцию
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ПОЛУширина
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    return a * (1 - np.abs((x - x0) / w)) * rect_1d(x, w=2 * w, x0=x0)


def gauss_1d(
        x: np.ndarray,
        a: Union[float, int] = 1.,
        w: Union[float, int] = 1.,
        x0: Union[float, int] = 0.,
        y0: Union[float, int] = 0.
) -> np.ndarray:
    """
    1D гауссоида
    :param x: np.ndarray координатная сетка
    :param a: амплитуда
    :param w: ширина по уровню a*e^-2
    :param x0: смещение по оси X (влево-вправо)
    :param y0: смещение по оси Y (вниз-вверх)
    :return: np.ndarray
    """
    return a * np.exp(-(x - x0) ** 2 / (w ** 2 / 8)) + y0


def gauss_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0.):
    """
    Возвращает 2-мерную гауссоиду с явно указанной амплитудой
    :param x: np.ndarray 2-мерная координатная сетка по оси X
    :param y: np.ndarray 2-мерная координатная сетка по оси Y
    :param a: Union[float, int] амплитуда
    :param wx: Union[float, int] ширина по оси X (может выступаить как СКО)
    :param wy: Union[float, int] ширина по оси Y (может выступаить как СКО)
    :param x0: Union[float, int] смещение относительно нуля координат по оси X
    :param y0: Union[float, int] смещение относительно нуля координат по оси Y
    :return: np.ndarray
    """
    return a * np.exp(-((x - x0) ** 2 / (wx ** 2 / 8) + (y - y0) ** 2 / (wy ** 2 / 8)))


def logistic_1d(x, a=1., w=1., x0=0.):
    """
    Возвращает 1-мерную логистическую функцию: https://en.wikipedia.org/wiki/Logistic_function
    :param x: np.ndarray координатная сетка
    :param a: Union[float, int] амплитуда
    :param w: Union[float, int] ширина
    :param x0: Union[float, int] смещение относительно нуля координат
    :return: np.ndarray
    """
    threshold = 70
    precision = 1e-10  # чтобы не хранить значения ~e-31 степени
    """
    Большие значения в степени exp приводят к переполнению, поэтому они отсекаются
    exp(70) = e+30 (30 знаков достаточно)
    exp(709) = e+307
    exp(710) = inf ~ overflow for np.float.64
    """
    k = 10 / w
    exp_term = -k * (x - x0)
    exp_term[exp_term > threshold] = threshold  # Отсечение больших значений в степени
    exp_term[exp_term < -threshold] = -threshold
    res = a / (1 + np.exp(exp_term))
    res[res < precision] = 0  # Отсечение очень маленьких значений в степени
    return res


def sin_1d(
        x: np.ndarray,
        a: Union[float, int] = 1.,
        x0: Union[float, int] = 0.,
        y0: Union[float, int] = 0.,
        T: Union[float, int] = 2 * np.pi,
        **kwargs
) -> np.ndarray:
    """
    1-мерная синусоида
    :param x: координатная сетка
    :param a: амплитуда
    :param x0: смещение по оси X
    :param y0: смещение по оси Y
    :param T: период
    :param clip: вырезать от left до right (default один период)
    :param left: default 0
    :param right: default T (вырезать полпериода right = T/2)
    :return:
    """
    result = a * np.sin( (x - x0) / (T / (2 * np.pi)) )

    clip = kwargs.get('clip', False)
    if clip:
        # get boundaries
        left = kwargs.get('left', 0)
        right = kwargs.get('right', T)
        # create masks
        left_mask = x - x0 > left
        right_mask = x - x0 < right
        # multiply masks
        result *= left_mask * right_mask

    return result + y0


def cos_1d(
        x: np.ndarray,
        a: Union[float, int] = 1.,
        x0: Union[float, int] = 0.,
        y0: Union[float, int] = 0.,
        T: Union[float, int] = 2 * np.pi,
        **kwargs
) -> np.ndarray:
    """
    1-мерная косинусоида
    :param x: координатная сетка
    :param a: амплитуда
    :param x0: смещение по оси X
    :param y0: смещение по оси Y
    :param T: период
    :param clip: вырезать от left до right (default один период)
    :param left: default 0
    :param right: default T (вырезать полпериода right = T/2)
    :return:
    """
    result = a * np.cos( (x - x0) / (T / (2 * np.pi)) )

    clip = kwargs.get('clip', False)
    if clip:
        # get boundaries
        left = kwargs.get('left', 0)
        right = kwargs.get('right', T)
        # create masks
        left_mask = x - x0 > left
        right_mask = x - x0 < right
        # multiply masks
        result *= left_mask * right_mask

    return result + y0


def semicircle(
        x: np.ndarray,
        r: Union[int, float] = 1,
        sag: Union[int, float] = None,
        x0: Union[int, float] = 0,
        y0: Union[int, float] = 0,
        inverse: Union[bool, int] = False,
):
    """
    Полуокружность
    :param x:
    :param r:
    :param sag: стрелка прогиба
    :param x0:
    :param y0:
    :param inverse:
    :return:
    """
    if r <= 0:
        raise ValueError(f'radius <= zero')

    # маска, чтобы отсечь nan
    mask = (r ** 2 - (x - x0) ** 2) < 0
    # уравнение полуокружности
    semicircle = np.sqrt(r ** 2 - (x - x0) ** 2)
    # сдвиг по оси y, чтобы получить нужное значение стрелки прогиба
    if sag:
        if sag > r:
            raise ValueError(f'sag {sag} greater than radius {r}')
        semicircle -= r - sag
        semicircle[semicircle < 0] = 0
    # приравниваем все nan к 0
    semicircle[mask] = 0
    # сдвиг по оси y
    semicircle += y0

    if inverse:
        semicircle = -semicircle

    return semicircle


def hemisphere(
        x: np.ndarray,
        y: np.ndarray,
        r: Union[int, float] = 1,
        sag: Union[int, float] = None,
        x0: Union[int, float] = 0,
        y0: Union[int, float] = 0,
        z0: Union[int, float] = 0,
        inverse: Union[bool, int] = False,
):
    """
    Полусфера
    :param x:
    :param y:
    :param r:
    :param sag: стрелка прогиба
    :param x0:
    :param y0:
    :param z0:
    :param inverse:
    :return:
    """
    if r <= 0:
        raise ValueError(f'radius <= zero')

    mask = (r ** 2 - (x - x0) ** 2 - (y - y0) ** 2) < 0
    hemisphere = np.sqrt(r ** 2 - (x - x0) ** 2 - (y - y0) ** 2)

    if sag:
        if sag > r:
            raise ValueError(f'sag {sag} greater than radius {r}')
        hemisphere -= r - sag
        hemisphere[hemisphere < 0] = 0

    hemisphere[mask] = 0
    hemisphere += z0

    if inverse:
        hemisphere = -hemisphere

    return hemisphere


def lens_1d(x, focus, wavelength, light_diameter, converge=True):
    k = 2 * np.pi / wavelength
    lens = k * np.sqrt(x ** 2 + focus ** 2)
    if converge:
        lens *= -1
    return lens * rect_1d(x, a=1, w=light_diameter)


def lens_2d(x, y, focus, wavelength, converge=True):
    k = 2 * np.pi / wavelength
    lens = k * np.sqrt(x ** 2 + y ** 2 + focus ** 2)
    if converge:
        lens *= -1
    return lens


def add_tilt_1d(x, complex_amplitude, wavelength, alpha, theta=0):
    """
    Added tilt to complex amplitude
    David Voelz. Computational Fourier Optics. A MATLAB® Tutorial. SPIE PRESS. p.89
    :param x:
    :param complex_amplitude: initial field
    :param wavelength: [meter]
    :param alpha: tilt angle [rad]
    :param theta: rotation along XY angle [rad]
    :return:
    """
    k = 2 * np.pi / wavelength
    tilt = x * np.cos(theta) * np.tan(alpha)
    return complex_amplitude * np.exp(1j * k * tilt)


def add_tilt_2d(x, y, complex_amplitude, wavelength, alpha, theta=0):
    """
    Added tilt to complex amplitude
    David Voelz. Computational Fourier Optics. A MATLAB® Tutorial. SPIE PRESS. p.89
    :param x:
    :param y:
    :param complex_amplitude: initial field
    :param wavelength: [meter]
    :param alpha: tilt angle [rad]
    :param theta: rotation along XY angle [rad]
    :return:
    """
    k = 2 * np.pi / wavelength
    tilt = (x * np.cos(theta) + y * np.sin(theta)) * np.tan(alpha)
    return complex_amplitude * np.exp(1j * k * tilt)
