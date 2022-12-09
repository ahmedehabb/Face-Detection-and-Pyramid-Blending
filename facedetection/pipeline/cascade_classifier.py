from .stage import Stage
from .feature import Feature

class Cascade_Classifier:
    def __init__(self, stages_lists : list[Stage], features_lists : list[Feature]) -> None:
        self.stages_lists = stages_lists
        self.features_lists = features_lists

        pass