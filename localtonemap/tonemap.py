"""
    Migrate from MATLAB's tonemap function
"""
import numpy as np
import localtonemap.util as util
import localtonemap.clahe as clahe

import matplotlib.pyplot as plt


def tonemap(E, l_remap=(0, 1), saturation=1., gamma_=1.5, numtiles=(4, 4)):
    """
        render HDR for viewing
        exposure estimate -> log2 -> CLAHE -> remap to l_remap -> gamma correction -> HDR
        @param E: exposure (N x M x 3)
        @param l_remap: remap intensity to l_remap in the image adjust step
        @param saturation: saturation of the color.
        @param numtiles: number of contextual tiles in the CLAHE step
        return contrast reduced image
    """
    if E.shape[0] % numtiles[0] != 0 or E.shape[1] % numtiles[1] != 0:
        E = util.crop_image(E, (E.shape[0] // numtiles[0] * numtiles[0], E.shape[1] // numtiles[1] * numtiles[1]))
    l2E, has_nonzero = lognormal(E)
    if has_nonzero:
        I = tone_operator(l2E, l_remap, saturation, gamma_, numtiles)
    else:
        I = l2E
    # clip
    I[I < 0] = 0
    I[1 < I] = 1
    return np.uint8(I * 255.)


def lognormal(E):
    """
        log2(E). remove 0s.
        return log2E, has_nonzero
    """
    mask = (E != 0)

    if np.any(mask):
        min_nonzero = np.min(E[mask])
        E[np.logical_not(mask)] = min_nonzero
        l2E = util.rescale(np.log2(E))
        has_nonzero = True

    else:# all elements are zero
        l2E = np.zeros_like(E)
        has_nonzero = False

    return l2E, has_nonzero

def tone_operator(l2E, l_remap, saturation, gamma_, numtiles):
    """
        The main algorithm is CLAHE: contrast limited adaptive histogram equalization
        preprocessing: convert RGB to XYZ to Lab
        postprocessing: back to RGB
    """
    lab = util.srgb2lab(l2E)
    lab[:,:,0] = util.rescale(lab[:,:,0])
#    lab[:, :, 0] /= 100
    lab[:, :, 0] = clahe.hist_equalize(lab[:, :, 0], numtiles)
    lab[:, :, 0] = imadjust(lab[:, :, 0], range_in=l_remap, range_out=(0, 1), gamma=gamma_) * 100
    lab[:, :, 1:] = lab[:, :, 1:] * saturation
    I = util.lab2srgb(lab)
    return I


def imadjust(I, range_in=None, range_out=(0, 1), gamma=1):
    """
        remap I from range_in to range_out
        @param I: image
        @param range_in: range of the input image. will be assigned minmax(I) if none
        @param range_out: range of the output image
        @param gamma: factor of the gamma correction
    """
    if range_in is None:
        range_in = (np.min(I), np.max(I))
    out = (I - range_in[0]) / (range_in[1] - range_in[0])
    out = out**gamma
    out = out * (range_out[1] - range_out[0]) + range_out[0]
    return out
