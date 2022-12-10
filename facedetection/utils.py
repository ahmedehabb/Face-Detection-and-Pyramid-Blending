import numpy as np


def integral_image(img):
    sum_rows = np.cumsum(img, axis=0)      # sum over rows for each of the columns
    return np.cumsum(sum_rows, axis=1)

