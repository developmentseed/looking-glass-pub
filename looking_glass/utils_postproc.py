"""
Collection of tools for post processing segmentation masks.

@author: Development Seed

utils_postproc.py
"""
import numpy as np


def get_thresh_weighted_sum(arr, thresh=0., weight=1.):
    """Helper for calculating total area from segmentation map.

    This function is useful for finding square meters of building or road area
    from a segmentation map. It finds all pixels meeting some threshold and
    then multiplies by a weight (one likely would use m^2 per pixel).

    Parameters
    ----------
    arr: np.ndarray
        Array to calculate sum of pixels meeting a threshold. For example,
        a segmentation map or stack of segmentation maps.
    thresh: float
        Threshold for pixel values. "Good" pixels are those greater or equal
        to the threshold. Default: 0.
    weight: float
        Multiplier for thresholded pixel values. Default: 1.

    Returns
    -------
    weighted_sum: float
        Number of pixels meeting `thresh` and multiplied by `weight`
    """

    # Error checking
    if not isinstance(arr, np.ndarray):
        raise ValueError(f'`arr` is not a numpy array. Got type:{type(arr)}')

    return weight * np.sum(arr >= thresh)
