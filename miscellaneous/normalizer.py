import numpy as np


def normalize(array: np.ndarray, **kwargs) -> np.ndarray:
    """
    Нормирует входной массив в диапазоне от new_min до new_max
    :param array:
    :param kwargs: old_min = min(array), old_max = max(array), new_min = 0., new_max = 1., dtype = np.float64
    :return:
    """
    if array.dtype in (np.complex64, np.complex128, np.csingle, np.cdouble, np.clongdouble):
        raise TypeError(f'Not implemented for complex-valued arrays: array.dtype = {array.dtype}')

    old_min = kwargs.get('old_min', np.min(array))
    old_max = kwargs.get('old_max', np.max(array))
    new_min = kwargs.get('new_min', 0.)
    new_max = kwargs.get('new_max', 1.)
    dtype = kwargs.get('dtype', np.float64)

    if old_max < old_min or new_max < new_min:
        raise ValueError(f'Значения максимумов должны превышать значения минимумов:'
                         f'old_min = {old_min}\nold_max = {old_max}\nnew_min = {new_min}\nnew_max = {new_max}')

    array = (array - old_min) / (old_max - old_min)  # from old_range to 0 ... 1.0
    array = array * (new_max - new_min) + new_min  # from 0 ... 1.0 to new_range

    return np.asarray(array, dtype=dtype)
