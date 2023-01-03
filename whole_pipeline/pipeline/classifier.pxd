from . cimport feature
import numpy as np
cimport numpy as np

cdef class Classifier_Stump:
    cdef public float left_node, right_node, node_threshold
    cdef public feature.Feature feature
    cdef public float classify_feature_summary(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale)
