"""
This file contains helper functions for calculating the SSIM, PSNR, and nRMSE metrics.
"""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def SSIM(rec, ref):
    """
    Get SSIM metrics for a reconstruction / reference a pair of slices
    """
    data_range = ref.max() - ref.min()
    ssim = structural_similarity(ref, rec, data_range=data_range)

    return ssim


def pSNR(rec, ref):
    """
    Get pSNR metrics for a reconstruction / reference a pair of slices
    """
    data_range = ref.max() - ref.min()
    psnr = peak_signal_noise_ratio(ref, rec, data_range=data_range)

    return psnr


def nRMSE(rec, ref):
    """
    Get normalized roort mean squared error for a reconstruction / reference a pair of slices

    """
    numerator = np.mean(np.abs(ref - rec))
    denominator = np.mean(np.abs(ref))
    return numerator / denominator
