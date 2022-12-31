import numpy as np
cimport numpy as np
from . cimport stage as stage_class, feature as feature_class
cdef bint boolean_variable = True


cdef class Cascade_Classifier:
    cdef public list features_lists
    cdef public list stages_lists
    cdef public int width, height
    cdef bint move_forward(self, int stage_index, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale)
    cdef public bint complete_pass(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale)

