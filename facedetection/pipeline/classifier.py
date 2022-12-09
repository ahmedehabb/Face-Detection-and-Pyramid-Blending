

class Classifier_Stump:
    def __init__(self, left_node : tuple(), right_node : tuple(), feature_idx : int, node_threshold : float ) -> None:
        self.node_threshold = node_threshold
        self.feature_idx = feature_idx
        self.left_node_index = left_node[0]
        self.left_node_value = left_node[1]
        self.right_node_index = right_node[0]
        self.right_node_value = right_node[1]
        return
    
        