from constants import WINDOW_SIZE
# from pipeline.cascade_classifier import Cascade_Classifier
import cv2
from skimage import io
import numpy as np
from parser1 import parse_haar_cascade_xml
from maincython import whole_loop

cascade  = parse_haar_cascade_xml()

img = io.imread('facedetection/data/images/company.jpeg',as_gray=True)
img = cv2.resize(img, (img.shape[1], img.shape[0]))

faces = whole_loop(img, cascade)
print(faces)
faces = cv2.groupRectangles(faces, 3, eps=0.1)[0]

for (x, y, x_end, y_end) in faces:
    img = cv2.rectangle(img, (x, y), (x_end, y_end), (1), 1)


io.imshow(img)
io.show()

