from .classifier import Classifier_Stump

class Stage:
    def __init__(self, classifiers: list[Classifier_Stump], stage_threshold: float) -> None:
        self.stage_threshold = stage_threshold
        self.classifiers = classifiers
        pass

    def test_stage(self, integral_image_window, variance, scale) -> bool:
        stage_summary : float = 0
        for classifier in self.classifiers: 
            stage_summary += classifier.classify_feature_summary(integral_image_window, variance, scale)

        # print(stage_summary, self.stage_threshold)
        return stage_summary > self.stage_threshold