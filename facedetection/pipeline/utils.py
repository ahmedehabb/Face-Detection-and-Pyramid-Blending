import numpy as np

def integral_image(image):
    """Compute the integral image of an image.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image.

    Returns
    -------
    integral_image : (M, N) ndarray
        Integral image.

    """
    return np.cumsum(np.cumsum(image, axis=1), axis=0)