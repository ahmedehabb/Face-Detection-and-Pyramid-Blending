from . cimport classifier
import numpy as np
cimport numpy as np
cdef bint boolean_variable = True


cdef class Stage:
    
    def __cinit__(self, list classifiers, float stage_threshold):
        self.stage_threshold = stage_threshold
        self.classifiers = classifiers
        pass

    cdef bint test_stage(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale):
        cdef float stage_summary = 0
        cdef classifier.Classifier_Stump classifier
        for classifier in self.classifiers: 
            stage_summary += classifier.classify_feature_summary(integral_image_window, variance, scale)

        # print(stage_summary, self.stage_threshold)
        return stage_summary > self.stage_threshold