import numpy as np
import localtonemap.util as util
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', size=30)

def hist_equalize(I, numtiles=(8, 8)):
    assert I.shape[0] % numtiles[0] == 0 and I.shape[1] % numtiles[1] == 0
    img_range = np.array([0, 1])
    tile_size = (I.shape[0] // numtiles[0], I.shape[1] // numtiles[1])
    tile_mappings = maketile_mapping(I, numtiles, tile_size, img_range, img_range)
    out = make_clahe_image(I, tile_mappings, numtiles, tile_size, img_range)
    return out


def maketile_mapping(I, numtiles, tile_size, selected_range, full_range, num_bins=256, norm_clip_limit=0.01):

    num_pixel_in_tile = np.prod(tile_size)
    min_clip_limit = np.ceil(np.float(num_pixel_in_tile) / num_bins)
    clip_limit = min_clip_limit + np.round(norm_clip_limit * (num_pixel_in_tile - min_clip_limit))

    tile_mappings = []
    # image_col = 0
    image_row = 0
    print('make tile mappings')

    for tile_row in range(numtiles[0]):
        tile_mappings.append([])
        image_col = 0
        # image_row = 0
        for tile_col in range(numtiles[1]):
            # print('tile ({}, {}):'.format(tile_row, tile_col), end=',')
            tile = I[image_row:(image_row + tile_size[0]), image_col:(image_col + tile_size[1])]
            # print('\timhist', end=',')
            tile_hist = imhist(tile, num_bins, full_range[1])

            # print('\tclip hist', end=',')
            tile_hist = clip_histogram(tile_hist, clip_limit, num_bins)

            """ plot histogram
            fig = plt.figure(figsize=(20, 12))
            plt.bar(np.arange(256) / 256., tile_hist, width=0.005, edgecolor='b');
            plt.xlim(0, 1);
            plt.xlabel('intensity');
            plt.ylabel('count');
            plt.tight_layout()
            plt.savefig('../result/intermediate/histogram/hist{}{}.pdf'.format(tile_row, tile_col));
            """

            # print('\tmake mapping')
            tile_mapping = make_mapping(tile_hist, selected_range, num_pixel_in_tile)
            tile_mappings[-1].append(tile_mapping)

            """ plot mapping
            fig = plt.figure(figsize=(20, 12))
            plt.plot(np.arange(256) / 256., tile_mapping, lw=2);
            plt.xlim(0, 1);
            plt.xlabel('x');
            plt.ylabel('f(x)');
            plt.tight_layout()
            plt.savefig('../result/intermediate/histogram/mapping{}{}.pdf'.format(tile_row, tile_col));
"""

            image_col += tile_size[1]
        image_row += tile_size[0]
    return tile_mappings


def imhist(tile, num_bins, top):
    """
        image histogram
        @param tile: a rectangular tile cropped from the image
        @param num_bins: number of bins
        @param top: scale the rightmost bin to the top
    """
    s = (num_bins - 1.) / top   # scale factor
    tile_scaled = np.floor(tile * s + .5)
    hist = np.zeros(num_bins, dtype=np.int32)
    for i in range(num_bins):
        hist[i] = np.sum(tile_scaled == i)
    return hist


def clip_histogram(img_hist, clip_limit, num_bins):
    """
        clip the histogram according to the clipLimit and redistributes clipped pixels across bins below the clipLimit
        @param img_hist: histogram of the image
        @param clip_limit: the clipping limit
        @param num_bins: number of bins
    """
    total_excess = np.sum(np.maximum(img_hist - clip_limit, 0))

    avg_bin_incr = np.floor(total_excess / num_bins)
    upper_limit = clip_limit - avg_bin_incr

    for k in range(num_bins):
        if img_hist[k] > clip_limit:
            img_hist[k] = clip_limit
        else:
            if img_hist[k] > upper_limit:
                total_excess -= clip_limit - img_hist[k]
                img_hist[k] = clip_limit
            else:
                total_excess -= avg_bin_incr
                img_hist[k] += avg_bin_incr

    # redistributes the remaining pixels, one pixel at a time
    k = 0
    # print('total excess={}'.format(total_excess), end=';')
    while total_excess != 0:
        step_size = max(int(np.floor(num_bins / total_excess)), 1)
        for m in range(k, num_bins, step_size):
            if img_hist[m] < clip_limit:
                img_hist[m] += 1
                total_excess -= 1
            if total_excess == 0:
                break

        k += 1
        if k == num_bins:
            k = 0
    return img_hist


def make_mapping(img_hist, selected_range, num_pixel_in_tile):
    """
        using uniform distribution
    """
    high_sum = np.cumsum(img_hist)
    val_spread = selected_range[1] - selected_range[0]

    scale = val_spread / num_pixel_in_tile
    mapping = np.minimum(selected_range[0] + high_sum * scale, selected_range[1])
    return mapping


def make_clahe_image(I, tile_mappings, numtiles, tile_size, selected_range, num_bins=256):
    """
        interpolates between neighboring tile mappings to produce a new mapping in order to remove artificially induced tile borders
    """
    assert num_bins > 1
    # print('make clahe image')
    Ic = np.zeros_like(I)

    bin_step = 1. / (num_bins - 1)
    start = np.ceil(selected_range[0] / bin_step)
    stop = np.floor(selected_range[1] / bin_step)

    aLut = np.arange(0, 1 + 1e-10, 1.0 / (stop - start))

    """ plot discontinuous
    imgtile_row = 0
    for tile_row in range(numtiles[0]):
        imgtile_col = 0
        for tile_col in range(numtiles[1]):
            mapping = tile_mappings[tile_row][tile_col]
            tile = I[imgtile_row:imgtile_row+tile_size[0], imgtile_col: imgtile_col+tile_size[1]];
            Ic[imgtile_row:imgtile_row+tile_size[0], imgtile_col: imgtile_col+tile_size[1]] = grayxform(tile, mapping);
            imgtile_col += tile_size[1]
        imgtile_row += tile_size[0]
    fig = plt.figure(figsize=(20, 12))
    plt.imshow(Ic, cmap='gray')
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    """

    imgtile_row = 0
    for k in range(numtiles[0] + 1):
        if k == 0:  # edge case: top row
            imgtile_num_rows = tile_size[0] // 2
            maptile_rows = (0, 0)
        elif k == numtiles[0]:
            imgtile_num_rows = tile_size[0] // 2
            maptile_rows = (numtiles[0] - 1, numtiles[0] - 1)
        else:
            imgtile_num_rows = tile_size[0]
            maptile_rows = (k - 1, k)

        imgtile_col = 0
        for l in range(numtiles[1] + 1):
            # print('tile ({}, {})'.format(k, l))
            if l == 0:
                imgtile_num_cols = tile_size[1] // 2
                maptile_cols = (0, 0)
            elif l == numtiles[1]:
                imgtile_num_cols = tile_size[1] // 2
                maptile_cols = (numtiles[1] - 1, numtiles[1] - 1)
            else:
                imgtile_num_cols = tile_size[1]
                maptile_cols = (l - 1, l)

            ul_maptile = tile_mappings[maptile_rows[0]][maptile_cols[0]]
            ur_maptile = tile_mappings[maptile_rows[0]][maptile_cols[1]]
            bl_maptile = tile_mappings[maptile_rows[1]][maptile_cols[0]]
            br_maptile = tile_mappings[maptile_rows[1]][maptile_cols[1]]

            norm_factor = imgtile_num_rows * imgtile_num_cols

            imgpxl_vals = grayxform(I[imgtile_row:(imgtile_row + imgtile_num_rows), imgtile_col:(imgtile_col + imgtile_num_cols)], aLut)

            row_w = np.tile(np.expand_dims(np.arange(imgtile_num_rows), axis=1), [1, imgtile_num_cols])
            col_w = np.tile(np.expand_dims(np.arange(imgtile_num_cols), axis=0), [imgtile_num_rows, 1])
            row_rev_w = np.tile(np.expand_dims(np.arange(imgtile_num_rows, 0, -1), axis=1), [1, imgtile_num_cols])
            col_rev_w = np.tile(np.expand_dims(np.arange(imgtile_num_cols, 0, -1), axis=0), [imgtile_num_rows, 1])

            Ic[imgtile_row:(imgtile_row + imgtile_num_rows), imgtile_col:(imgtile_col + imgtile_num_cols)] = (row_rev_w * (col_rev_w * grayxform(imgpxl_vals, ul_maptile) + col_w * grayxform(imgpxl_vals, ur_maptile)) + row_w * (col_rev_w * grayxform(imgpxl_vals, bl_maptile) + col_w * grayxform(imgpxl_vals, br_maptile))) / norm_factor

            imgtile_col += imgtile_num_cols

        imgtile_row += imgtile_num_rows
    return Ic


def grayxform(I, aLut):
    """
        map I to aLut
        @param I: image
        @param aLut: look-up table
    """
    max_idx = len(aLut) - 1
    val = np.copy(I)
    val[val < 0] = 0
    val[val > 1] = 1
    indexes = np.int32(val * max_idx + 0.5)
    return aLut[indexes]
