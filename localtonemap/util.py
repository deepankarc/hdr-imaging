import numpy as np

E = 216. / 24389
K = 24389. / 27

M = np.array([[0.412424, 0.212656, 0.0193324],
    [0.357579, 0.715158, 0.119193],
    [0.180464, 0.0721856, 0.950444]])


def hist_count(I):
    """
        count [0,255]
        @param I: image
    """
    freq = np.zeros(256, dtype=int)
    for x in I.ravel():
        freq[x] += 1
    return freq


def rescale(I, window=(0, 1)):
    """
       rescale the intensity of the image [Imin, Imax] -> window
       @param window: tuple 1x2
    """
    a = np.min(I)
    b = np.max(I)

    J = I
    if a == b:
        if a == 0:
            J = I
        else:
            J = I / a
    else:
        J = (I - a) / (b - a) * (window[1] - window[0]) + window[0]
    return J


# image channel conversion
def srgb2lab(rgb):
    dims = rgb.shape
    rgb = np.reshape(np.transpose(rgb, (2, 0, 1)), [3, dims[0] * dims[1]])
    lab_1d = xyz2lab(srgb2xyz(rgb))
    lab = np.transpose(np.reshape(lab_1d, [3, dims[0], dims[1]]), (1, 2, 0))
    return lab


def srgb2xyz(rgb):
    """
        http://brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    """
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[np.logical_not(mask)] = rgb[np.logical_not(mask)] / 12.92

    return np.dot(M.T, rgb)


def xyz2lab(xyz):
    """
        http://brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
    """

    xyz[0, :] = xyz[0, :] / 0.95047
    xyz[2, :] = xyz[2, :] / 1.08883

    mask = xyz > E

    xyz[mask] = xyz[mask] ** (1. / 3)
    xyz[np.logical_not(mask)] = (K * xyz[np.logical_not(mask)] + 16) / 116.

    lab = np.zeros_like(xyz)
    lab[0, :] = 116. * xyz[1, :] - 16.
    lab[1, :] = 500. * (xyz[0, :] - xyz[1, :])
    lab[2, :] = 200. * (xyz[1, :] - xyz[2, :])
    return lab


def lab2srgb(lab):
    dims = lab.shape
    lab = np.reshape(np.transpose(lab, (2, 0, 1)), [3, dims[0] * dims[1]])
    rgb = xyz2srgb(lab2xyz(lab))
    rgb = np.transpose(np.reshape(rgb, [3, dims[0], dims[1]]), (1, 2, 0))
    return rgb


def lab2xyz(lab):
    """
        http://brucelindbloom.com/index.html?Eqn_Lab_to_XYZ.html
    """
    f = np.copy(lab)
    xyz = np.copy(f)

    mask = lab[0, :] > K * E
    mask_not = np.logical_not(mask)
    xyz[1, mask] = ((lab[0, mask] + 16.0) / 116.) ** 3
    xyz[1, mask_not] = lab[0, mask_not] / K

    mask = xyz[1, :] > E
    mask_not = np.logical_not(mask)
    f[1, mask] = (lab[0, mask] + 16.0) / 116.
    f[1, mask_not] = (K * xyz[1, mask_not] + 16.) / 116.

    f[0, :] = lab[1, :] / 500. + f[1, :]
    f[2, :] = f[1, :] - lab[2, :] / 200.

    tmp = f[0, :] ** 3
    mask = tmp > E
    mask_not = np.logical_not(mask)
    xyz[0, mask] = tmp[mask]
    xyz[0, mask_not] = (116. * f[0, mask_not] - 16.) / K

    tmp = f[2, :] ** 3
    mask = tmp > E
    mask_not = np.logical_not(mask)
    xyz[2, mask] = tmp[mask]
    xyz[2, mask_not] = (116. * f[2, mask_not] - 16.) / K

    xyz[0, :] = xyz[0, :] * 0.95047
    xyz[2, :] = xyz[2, :] * 1.08883
    return xyz


def xyz2srgb(xyz):
    """
        http://brucelindbloom.com/index.html?Eqn_XYZ_to_RGB.html
    """
    invMT = np.linalg.inv(M.T)

    rgb = np.dot(invMT, xyz)
    mask = rgb > 0.0031308
    mask_not = np.logical_not(mask)
    rgb[mask] = ((1.055 * rgb[mask]) ** (1 / 2.4)) - 0.055
    rgb[mask_not] = 12.92 * rgb[mask_not]
    return rgb


def crop_image(I, crop_size):
    return I[max((I.shape[0] - crop_size[0]) // 2, 0):min((I.shape[0] + crop_size[0]) // 2, I.shape[0]), max((I.shape[1] - crop_size[1]) // 2, 0):min((I.shape[1] + crop_size[1]) // 2, I.shape[1])]
