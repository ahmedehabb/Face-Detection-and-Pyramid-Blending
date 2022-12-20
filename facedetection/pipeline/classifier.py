from .feature import Feature


class Classifier_Stump:
    def __init__(self, left_node : float, right_node : float, feature : Feature, node_threshold : float ) -> None:
        self.node_threshold = node_threshold
        self.left_node_value = left_node
        self.feature = feature
        self.right_node_value = right_node
        return

    def classify_feature_summary(self, integral_image_window, variance, scale):
        
        # get feature value
        feature_value = self.feature.compute_feature(integral_image_window, scale)
        
        # All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions
        # During scanning the effect of image normalization can be achieved by post-multiplying the feature values rather than pre-multiplying the pixels.
        # so we multiply node_threshold at end of stage summary instead of doing preprocessing on integral window
        # equivalent to dividing value by variance ~
        # print(feature_value , self.node_threshold * variance)
        if feature_value < (self.node_threshold * variance):
            return self.left_node_value 
        return self.right_node_value
    
        