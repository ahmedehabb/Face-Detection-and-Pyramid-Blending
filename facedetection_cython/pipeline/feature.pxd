import numpy as np
cimport numpy as np
from . cimport rectangle


cdef class Feature:
    cdef:
        list rect_lists
    cdef public float compute_feature(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, int scale)
