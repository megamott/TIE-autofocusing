import math
import numpy as np


def central_finite_difference(planes: tuple, h: float = 1., deriv_order: int = 1) -> np.ndarray:
    coefs = coefficients_grid(deriv_order, len(planes))
    coefs = np.delete(coefs, len(coefs)//2)

    result = np.array([plane * coefs[count] / h for count, plane in enumerate(planes)])
    return np.sum(result, axis=0)


def coefficients_grid(deriv_order: int, accuracy: int) -> np.ndarray:
    num_central = 2 * math.floor((deriv_order + 1) / 2) - 1 + accuracy
    num_side = num_central // 2
    offsets = list(range(-num_side, num_side + 1))

    center = _calc_coefs(deriv_order, offsets)
    return center


def _build_rhs(offsets: list, deriv: int) -> np.ndarray:
    b = [0 for _ in offsets]
    b[deriv] = math.factorial(deriv)

    return np.array(b, dtype='float')


def _build_matrix(offsets: list) -> np.ndarray:
    a = [([1 for _ in offsets])]
    for i in range(1, len(offsets)):
        a.append([j ** i for j in offsets])

    return np.array(a, dtype='float')


def _calc_coefs(deriv: int, offsets: list) -> np.ndarray:
    matrix = _build_matrix(offsets)
    rhs = _build_rhs(offsets, deriv)

    coefs = np.linalg.solve(matrix, rhs)
    return coefs
