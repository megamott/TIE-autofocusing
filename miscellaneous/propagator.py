import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


def rect_1d(x, a=1., w=1., x0=0., y0=0):
    return a * (np.abs((x - x0) / w) < 0.5) + y0


def rect_2d(x, y, a=1., wx=1., wy=1., x0=0., y0=0., z0=0):
    return a * (rect_1d(x, w=wx, x0=x0) * rect_1d(y, w=wy, x0=y0)) + z0


def angular_spectrum_band_limited(
        complex_field: np.ndarray,
        distance: float,
        wavelength: float,
        px_size: float
) -> np.ndarray:
    # Увеличение транспаранта в 2 раза для трансформации линейной свертки в циклическую
    # (периодические граничные условия)
    if complex_field.ndim == 2:
        height = 2 * complex_field.shape[0]
        width = 2 * complex_field.shape[1]

        # Индексы для "старого" поля
        left = int(width * .25)
        right = int(width * .75)
        top = int(height * .25)
        bottom = int(height * .75)

        # Вписываем "старое" поле в новое
        new_field = np.zeros((height, width), dtype=complex_field.dtype)
        new_field[top:bottom, left:right] = complex_field

        # Сетка в частотной области
        nu_x = np.arange(-width / 2, width / 2) / (width * px_size)
        nu_y = np.arange(-height / 2, height / 2) / (height * px_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)
        nu_x_grid, nu_y_grid = ifftshift(nu_x_grid), ifftshift(nu_y_grid)
        nu_z_grid = np.sqrt(wavelength ** -2 - nu_x_grid ** 2 - nu_y_grid ** 2)
        nu_z_grid[nu_x_grid ** 2 + nu_y_grid ** 2 > wavelength ** -2] = 0

        # Расчет граничных частот U/V_limit
        dnu_x = 1 / (width * px_size)
        dnu_y = 1 / (height * px_size)
        nu_x_limit = 1 / (np.sqrt((2 * dnu_x * distance) ** 2 + 1) * wavelength)
        nu_y_limit = 1 / (np.sqrt((2 * dnu_y * distance) ** 2 + 1) * wavelength)

        # Передаточная функция (угловой спектр)
        h_clipper = rect_2d(nu_x_grid, nu_y_grid, wx=2 * nu_x_limit, wy=2 * nu_y_limit)
        h = np.exp(1j * 2 * np.pi * nu_z_grid * distance) * h_clipper

        # обратное преобразование Фурье
        return ifft2(fft2(new_field) * h)[top:bottom, left:right]
