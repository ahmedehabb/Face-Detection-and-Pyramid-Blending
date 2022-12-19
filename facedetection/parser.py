import xml.etree.ElementTree as ET

from pipeline.classifier import Classifier_Stump
from pipeline.stage import Stage
from pipeline.rectangle import Rectangle
from pipeline.feature import Feature
from pipeline.cascade_classifier import Cascade_Classifier

def parse_haar_cascade_xml(xml_file: str = "facedetection/data/haarcascades/haarcascade_frontalface_default.xml") -> Cascade_Classifier :
    
    # reads the xml features and stages file from opencv pretrained models
    all = ET.parse(xml_file)
    cascade = all.find("cascade")
    
    width , height = int(cascade.find("width").text), int(cascade.find("height").text)
    stages = cascade.find("stages")
    features = cascade.find("features")
    stages_list, features_list = [], []

    # will read features first since will populate them in the classifiers
    for feature in features:

        rects = feature.find("rects")
        rects_list = []

        for rect in rects:
            # It obviously describes parameters of rectangle (x, y, width, height) and the weight of rectangle. 
            x, y, rect_width, rect_height, rect_weight = map(float, (rect.text).split())
            x, y, rect_width, rect_height = int(x), int(y), int(rect_width), int(rect_height)
            rects_list.append(Rectangle(x, y, rect_width, rect_height, rect_weight))
        
        features_list.append(Feature(rects_list))

    for stage in stages:

        classifiers = stage.find("weakClassifiers")
        stage_threshold = float(stage.find("stageThreshold").text)
        classifiers_list = []
        for classifier in classifiers:

            internal_nodes = map(float, (classifier.find("internalNodes").text).split())
            leaf_values = map(float, (classifier.find("leafValues").text).split())

            # skipping left node and right node indexes
            _, _, feature_idx, node_threshold = internal_nodes
            feature_idx = int(feature_idx)
            left_node_val, right_node_val = leaf_values
            classifiers_list.append(Classifier_Stump(left_node_val, right_node_val, features_list[feature_idx], node_threshold))

        stages_list.append(Stage(classifiers_list, stage_threshold))
    


    return Cascade_Classifier(stages_list, features_list, width, height)

# cascade : Cascade_Classifier = parse_haar_cascade_xml()


