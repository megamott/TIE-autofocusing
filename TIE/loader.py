import numpy as np
from typing import List


def load_files(paths: List[str]) -> List[np.ndarray]:
    return [load_file(path) for path in paths]


def load_file(path: str) -> np.ndarray:
    array = np.load(path)
    return array
