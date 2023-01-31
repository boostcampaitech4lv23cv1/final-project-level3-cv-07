import math
import numpy as np

def calc_euclidean_dist(x, y, cx, cy):
    return math.sqrt((cx - x) ** 2 + (cy - y) ** 2)


def calc_manhattan_dist(x, y, cx, cy):
    return abs(cx - x) + abs(cy - y)


# mask generator v0 (exactly same as not using mask)
def mask_generator_v0(x_min, y_min, x_max, y_max):
    w = x_max - x_min
    h = y_max - y_min
    mask = np.ones(shape=(h, w), dtype=np.float16)
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


# mask generator v1 (using Euclidean distance and thresholding)
def mask_generator_v1(x_min, y_min, x_max, y_max, thr=0.7):
    w = x_max - x_min
    h = y_max - y_min
    cx = w // 2
    cy = h // 2
    mask = np.zeros(shape=(h, w), dtype=np.float16)
    max_dist = calc_euclidean_dist(0, 0, cx, cy)

    # fill mask with L2 distance (from each pixel to center pixel)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            mask[i, j] = calc_euclidean_dist(j, i, cx, cy)
    mask /= max_dist  # normalize all dist
    mask = 1 - mask
    mask[mask >= thr] = 1
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


# mask generator v2 (using Manhattan distance)
def mask_generator_v2(x_min, y_min, x_max, y_max, thr=0.7):
    w = x_max - x_min
    h = y_max - y_min
    cx = w // 2
    cy = h // 2
    mask = np.zeros(shape=(h, w), dtype=np.float16)
    max_dist = calc_manhattan_dist(0, 0, cx, cy)

    # fill mask with L1 distance (from each pixel to center pixel)
    for i in range(len(mask)):
        for j in range(len(mask[0])):
            mask[i, j] = calc_manhattan_dist(j, i, cx, cy)
    mask /= max_dist  # normalize all dist
    mask = 1 - mask
    mask[mask >= thr] = 1
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask


# mask generator v3 (using padding)
def mask_generator_v3(x_min, y_min, x_max, y_max, level=10, step=3):
    w = x_max - x_min
    h = y_max - y_min

    n_w = w - level * 2 * step
    n_h = h - level * 2 * step

    if n_w <= 0 or n_h <= 0:
        mask = np.ones(shape=(h, w), dtype=np.float16)
        mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
        return mask, 1 - mask

    mask = np.ones(shape=(n_h, n_w), dtype=np.float16)

    for i in range(level):
        const = 1 - (1 / level * (i + 1))
        mask = np.pad(
            mask, ((step, step), (step, step)), "constant", constant_values=const
        )
    mask = np.reshape(np.repeat(mask, 3), (h, w, 3))
    return mask, 1 - mask
