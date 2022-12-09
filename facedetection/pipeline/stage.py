from .classifier import Classifier_Stump

class Stage:
    def __init__(self, classifiers: list[Classifier_Stump], stage_threshold: float) -> None:
        self.stage_threshold = stage_threshold
        self.classifiers = classifiers
        pass