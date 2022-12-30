import numpy as np
cimport numpy as np
from . cimport classifier
cdef bint boolean_variable = True


cdef class Stage:
    cdef:
        list classifiers
        float stage_threshold
    cdef public bint test_stage(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale)
