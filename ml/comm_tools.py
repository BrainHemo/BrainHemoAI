import numpy as np
from skimage import transform
import typing

default_windows = [[0, 100], [10, 90], [20, 80], [30, 80], [40, 80]]


def crop_data(data: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    assert data.ndim == 3
    coords = np.array(np.nonzero(data != 0))
    top_left: np.ndarray = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    cropped_data = data[:, top_left[-2]:bottom_right[-2], top_left[-1]:bottom_right[-1]]
    return cropped_data, top_left[-2:]


def resize_data(data: np.ndarray, resize_shape=(352, 288)) -> np.ndarray:
    assert data.ndim == 3
    dims = data.shape[0]
    result = np.zeros(shape=(dims, *resize_shape), dtype=data.dtype)
    for idx in range(len(data)):
        result[idx] = transform.resize(data[idx], resize_shape, preserve_range=True)
    result = result.astype(np.float32)
    return result


def resize_each_blood_type(data_blood: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """
    resize the blood one by one to avoid interpolation.
    """
    assert data_blood.ndim == 3
    result = np.zeros(shape=(data_blood.shape[0], *shape), dtype=data_blood.dtype)
    for type_val in np.unique(data_blood):
        if type_val == 0:
            continue
        for i in range(data_blood.shape[0]):
            one_type = np.zeros_like(data_blood[i])
            one_type[data_blood[i] == type_val] = 1
            one_type = transform.resize(one_type, shape, preserve_range=True)
            result[i][one_type > 0.5] = type_val
    return result


def area_clip(data: np.ndarray, min_size=0.2):
    """
    remove the slice which with small brain tissue
    """
    nz, nx, ny = data.shape

    start = 0
    end = nz - 1
    foreground = data != 0
    min_size = min_size * nx * ny
    while start < nz and np.sum(foreground[start]) < min_size:
        start += 1
    while end >= 0 and np.sum(foreground[end]) < min_size:
        end -= 1

    return start, end


def pad_zdims(data: np.ndarray, z: int) -> np.ndarray:
    """
    keep the number of slice to z
    """
    nz = data.shape[0]
    if nz >= z:
        return data[:z]
    else:
        shape = list(data.shape)
        shape[0] = z
        result = np.zeros(shape, dtype=data.dtype)
        result[:nz] = data
        return result


def get_win_data(data: np.ndarray, wins=None, norm=True) -> np.ndarray:
    if wins is None:
        wins = default_windows
    windows_mask = [np.logical_and(data > x, data < y) for x, y in wins]
    # normalize
    if norm:
        foreground = data != 0
        avg = np.average(data[foreground])
        std = np.std(data[foreground])
        data = (data - avg) / std
    # compose the multi-channel data
    result = []
    for mask in windows_mask:
        temp = np.zeros_like(data)
        temp[mask] = data[mask]
        result.append(temp)

    return np.array(result, dtype=np.float32)
