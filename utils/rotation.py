import typing

import numpy as np
from skimage import transform


def rotate(data: np.ndarray, angle: float, center: typing.Tuple) -> np.ndarray:
    _data = data.copy()
    if data.ndim == 2:
        _data = np.expand_dims(_data, axis=0)
    nz = len(_data)
    result = np.zeros_like(_data)
    for z in range(nz):
        result[z] = transform.rotate(_data[z], angle=angle, center=center, preserve_range=True)
    return result
