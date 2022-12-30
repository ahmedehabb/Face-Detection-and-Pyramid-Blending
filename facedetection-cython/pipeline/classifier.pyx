from . cimport feature
import numpy as np
cimport numpy as np

cdef class Classifier_Stump:
    
    def __cinit__(self, left_node : float, right_node : float, feature : feature.Feature, node_threshold : float ):
        self.node_threshold = node_threshold
        self.left_node = left_node
        self.feature = feature
        self.right_node = right_node
        return

    cdef float classify_feature_summary(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale):
        
        # get feature value
        cdef float feature_value = self.feature.compute_feature(integral_image_window, scale)
        
        # All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions
        # During scanning the effect of image normalization can be achieved by post-multiplying the feature values rather than pre-multiplying the pixels.
        # so we multiply node_threshold at end of stage summary instead of doing preprocessing on integral window
        # equivalent to dividing value by variance ~
        # print(feature_value , self.node_threshold * variance)
        if feature_value < (self.node_threshold * variance):
            return self.left_node 
        return self.right_node
    
        