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

    def move_forward(self, stage_index) -> bool:
        current_stage : Stage = self.stages_lists[stage_index]
        current_stage_classifiers : list[Classifier_Stump] = current_stage.classifiers
        stage_summary : float = 0
        for classifier in current_stage_classifiers: 
            # compute feature
            feature = self.features_lists[classifier.feature_idx]
            # get value
            value = 0 

            if value <  classifier.node_threshold:
                stage_summary += classifier.left_node_value
            else:
                stage_summary += classifier.right_node_value

        return stage_summary >= current_stage.stage_threshold

    def complete_pass(self) -> bool:
        for stage_index in range(len(self.stages_lists)):
            if self.move_forward(stage_index) == False:
                return False

        return True

            

