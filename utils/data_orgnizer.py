from typing import Tuple

import SimpleITK
import numpy as np
from numpy import ndarray
from skimage import transform, morphology


def resample_image(image, spacing=(0.5, 0.5, 5), method=SimpleITK.sitkNearestNeighbor) -> SimpleITK.Image:
    """
    resample the spacing.
    this is optional.
    """
    org_spacing = np.array(image.GetSpacing())

    spacing = np.array(spacing)
    org_size = np.array(image.GetSize())
    dst_size = org_size * (org_spacing / spacing)
    dst_size = np.round(dst_size).astype(int)

    resampler = SimpleITK.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(dst_size.tolist())
    resampler.SetOutputSpacing(spacing)
    resampler.SetTransform(SimpleITK.Transform(3, SimpleITK.sitkIdentity))
    resampler.SetInterpolator(method)

    result = resampler.Execute(image)

    return result


def rectify_brain_mask(head: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:

    foreground = np.logical_and(brain_mask != 0, np.logical_and(head >= -20, head <= 100))

    mask = np.zeros_like(foreground, dtype=bool)
    for idx in range(len(foreground)):
        temp = foreground[idx].copy()
        temp = morphology.remove_small_holes(temp, area_threshold=10000)
        temp = morphology.remove_small_objects(temp, min_size=1000)
        mask[idx] = temp

    return mask


def rotate_each_blood_type(data_blood: np.ndarray, angle: float, center) -> np.ndarray:
    """
    resize the blood one by one to avoid interpolation.
    """
    assert data_blood.ndim == 3

    result = np.zeros_like(data_blood)
    for z in range(len(data_blood)):
        for type_val in np.unique(data_blood):
            if type_val == 0:
                continue
            one_type = np.zeros_like(data_blood[z])
            one_type[data_blood[z] == type_val] = 1
            one_type = transform.rotate(one_type, angle=angle, center=center, preserve_range=True)
            result[z][one_type > 0.5] = type_val
    return result


def prepare_data(head: np.ndarray, m_brain: np.ndarray, rotation=None) -> Tuple[ndarray, ndarray]:
    m_brain = rectify_brain_mask(head, m_brain)
    # Remove calcified regions
    m_brain = np.logical_and(m_brain, head < 120)
    if rotation is not None:
        angle, center = rotation
        for idx in range(len(m_brain)):
            head[idx] = transform.rotate(head[idx], angle=angle, center=center, preserve_range=True)
            m_brain[idx] = transform.rotate(m_brain[idx], angle=angle, center=center) > 0.5

    return head, m_brain

