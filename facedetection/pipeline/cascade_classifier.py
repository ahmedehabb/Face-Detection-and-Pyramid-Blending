from .stage import Stage
from .feature import Feature
from .classifier import Classifier_Stump

class Cascade_Classifier:
    def __init__(self, stages_lists : list[Stage], features_lists : list[Feature], width : int, height : int) -> None:
        self.stages_lists = stages_lists
        self.features_lists = features_lists
        self.width = width
        self.height = height
        pass

    def move_forward(self, stage_index, integral_image_window, variance, scale) -> bool:
        current_stage : Stage = self.stages_lists[stage_index]
        return current_stage.test_stage(integral_image_window, variance, scale)

    def complete_pass(self, integral_image_window, variance, scale) -> bool:
        
        for stage_index in range(len(self.stages_lists)):
            if self.move_forward(stage_index,integral_image_window, variance, scale) == False:
                return False
        
        print(stage_index)
        return True

            

