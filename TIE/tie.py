import numpy as np

from typing import Tuple, Optional, List
from numpy.fft import fftfreq, fft2, ifft2
from abc import ABC, abstractmethod

from finite_difference import central_finite_difference


class TIESolver(ABC):
    def __init__(
            self,
            intensities: List[np.ndarray],
            dz: float,
            wavelength: Optional[float],
    ):
        if len(intensities) > 2:
            raise NotImplementedError(f'Expect 2 intensities, instead got {len(intensities)}')

        self.intensities = intensities
        self.dz = dz
        self.wavelength = wavelength
        self.axial_derivative = central_finite_difference(self.intensities, dz/2)

        self.ref_intensity = self.intensities[0]

    @abstractmethod
    def solve(self, threshold) -> np.ndarray:
        pass

    def add_threshold(self, threshold: float):
        threshold *= np.max(self.ref_intensity)
        mask = self.ref_intensity < threshold
        self.ref_intensity[mask] = threshold
        return mask


class FFTSolver2D(TIESolver):
    def __init__(self, intensities, dz, wavelength, pixel_size):
        super().__init__(intensities, dz, wavelength)
        self.pixel_size = pixel_size
        self.kx, self.ky = self._get_frequency_coefs()

    def solve(self, threshold) -> np.ndarray:
        wave_number = 2 * np.pi / self.wavelength
        phase = - wave_number * self.axial_derivative

        # 1. Обратный Лапласиан
        zero_k_mask = (self.kx == 0) & (self.ky == 0)
        self.kx[zero_k_mask] = 1. + 0*1j
        self.ky[zero_k_mask] = 1. + 0*1j
        phase = fft2(phase)
        phase = phase / (self.kx ** 2 + self.ky ** 2)
        phase[zero_k_mask] = 0. + 0*1j
        self.kx[zero_k_mask] = 0. + 0*1j
        self.ky[zero_k_mask] = 0. + 0*1j

        # 2. Градиенты
        phase_x = ifft2(phase * self.kx).real
        phase_y = ifft2(phase * self.ky).real

        # 3. Деление на опорную интенсивность
        mask = self.add_threshold(threshold)
        phase_x /= self.ref_intensity
        phase_y /= self.ref_intensity
        phase_x[mask] = 0
        phase_y[mask] = 0

        # 4. Градиент
        phase_x = fft2(phase_x) * self.kx
        phase_y = fft2(phase_y) * self.ky

        # 5. Обратный Лапласиан
        self.kx[zero_k_mask] = 1. + 0*1j
        self.ky[zero_k_mask] = 1. + 0*1j
        phase_x = phase_x / (self.kx ** 2 + self.ky ** 2)
        phase_y = phase_y / (self.kx ** 2 + self.ky ** 2)
        phase_x[zero_k_mask] = 0. + 0*1j
        phase_y[zero_k_mask] = 0. + 0*1j
        self.kx[zero_k_mask] = 0. + 0*1j
        self.ky[zero_k_mask] = 0. + 0*1j
        phase_x = ifft2(phase_x).real
        phase_y = ifft2(phase_y).real

        phase = phase_x + phase_y

        return phase

    def _get_frequency_coefs(self) -> Tuple[np.ndarray, np.ndarray]:
        h, w = self.ref_intensity.shape
        nu_x = fftfreq(w, d=self.pixel_size)
        nu_y = fftfreq(h, d=self.pixel_size)
        nu_x_grid, nu_y_grid = np.meshgrid(nu_x, nu_y)

        kx = 1j * 2 * np.pi * nu_x_grid
        ky = 1j * 2 * np.pi * nu_y_grid

        return kx, ky



