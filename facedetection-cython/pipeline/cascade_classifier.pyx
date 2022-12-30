from . cimport stage as stage_class, feature as feature_class
import numpy as np
cimport numpy as np
cdef bint boolean_variable = True


cdef class Cascade_Classifier:
   
    def __cinit__(self, list stages_lists, list features_lists, int width, int height):
        self.stages_lists = stages_lists
        self.features_lists = features_lists
        self.width = width
        self.height = height
        pass

    cdef bint move_forward(self, int stage_index, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale):
        cdef stage_class.Stage current_stage = self.stages_lists[stage_index]
        return current_stage.test_stage(integral_image_window, variance, scale)

    cdef bint complete_pass(self, np.ndarray[np.float64_t, ndim=2] integral_image_window, float variance, int scale):
        
        for stage_index in range(len(self.stages_lists)):
            if self.move_forward(stage_index,integral_image_window, variance, scale) == False:
                return False
        return True

            

