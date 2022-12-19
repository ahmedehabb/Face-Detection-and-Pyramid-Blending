from parser import parse_haar_cascade_xml
from constants import PADDING, WINDOW_SIZE, SCALE_FACTOR, STARTING_SCALE, DELTA_SHIFT

from utils import integral_image, integral_of_image_squared, variance
from pipeline.cascade_classifier import Cascade_Classifier
from skimage import io,draw
from skimage.feature import canny
from skimage.filters.edges import sobel_h,sobel_v
from skimage.exposure import histogram
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import cv2

cascade: Cascade_Classifier = parse_haar_cascade_xml()
img = io.imread('facedetection/data/images/company.jpeg', as_gray=True)
img_2 = np.copy(img)

img_width, img_height = img.shape[1], img.shape[0]
# normalizing image mean and variance
img_normalized = np.copy(img)
mean = np.mean(img_normalized)
img_normalized = img_normalized - mean
# All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions.
img_normalized = img_normalized / np.sqrt(np.var(img_normalized))
img = img_normalized
print(np.mean(img_normalized), np.var(img_normalized))
# edge_strength = abs(sobel_h(img) + sobel_v(img))
# edge_strength = edge_strength / np.max(edge_strength)
# edge_strength_integral = integral_image(edge_strength)


# defining the window size specs as in constants
window_size = WINDOW_SIZE


# getting the integral image and integral imag squared
integral_image = integral_image(img)
integral_of_image_squared = integral_of_image_squared(img)


faces = []

scale = STARTING_SCALE
max_scale = 15
while scale < max_scale:
    print(scale)
    # define current scale and window size accordingly
    current_window_size = int(window_size[0] * scale), int(window_size[1] * scale)

    # defining integral window size to be window size + padding added to correctly index integral window
    integral_window_size = current_window_size[0] + PADDING ,current_window_size[1] + PADDING


    for x in range(0, img_width - current_window_size[1] + 1, int(np.ceil(scale * DELTA_SHIFT))):
        for y in range(0, img_height - current_window_size[0] + 1, int(np.ceil(scale * DELTA_SHIFT))):
            # window = img[y : y+window_size[0], x : x+window_size[1]]
            # computing the integral window of the current x,y window
            integral_window = integral_image[y : y +  integral_window_size[0] , x : x + integral_window_size[1]]
            # computing the integral window of the current x,y window
            integral_window_squared = integral_of_image_squared[y : y +  integral_window_size[0] , x : x + integral_window_size[1]]
            # computing the variance
            integral_window_variance = np.sqrt(variance(integral_window, integral_window_squared))
            if integral_window_variance < 0.25:
                continue
            # integral_window_variance = 1
            
            # print(integral_window_variance)
            if(cascade.complete_pass(integral_window, integral_window_variance, scale)):
                print("complete", integral_window_variance)
                faces.append((x,y,scale))

    
    # increasing the scale by factor : SCALE_FACTOR
    scale = int(np.ceil(SCALE_FACTOR * scale))


print(faces)
# faces = [(38, 189, 1), (321, 175, 1), (6, 114, 1), (33, 179, 1), (62, 176, 1), (63, 176, 1), (64, 175, 1), (65, 175, 1), (85, 188, 1), (86, 187, 1), (86, 189, 1), (87, 188, 1), (87, 189, 1), (88, 188, 1), (88, 189, 1), (113, 170, 1), (113, 171, 1), (114, 171, 1), (169, 168, 1), (169, 169, 1), (169, 170, 1), (170, 168, 1), (170, 169, 1), (170, 170, 1), (171, 170, 1), (192, 187, 1), (192, 188, 1), (192, 189, 1), (192, 190, 1), (193, 187, 1), (193, 188, 1), (193, 189, 1), (193, 191, 1), (194, 187, 1), (194, 188, 1), (194, 189, 1), (194, 190, 1), (195, 188, 1), (195, 189, 1), (203, 39, 1), (229, 167, 1), (257, 185, 1), (258, 185, 1), (258, 186, 1), (259, 185, 1), (260, 185, 1), (272, 168, 1), (303, 185, 1), (303, 186, 1), (304, 185, 1), (304, 186, 1), (305, 185, 1), (305, 186, 1), (318, 171, 1), (319, 171, 1), (320, 172, 1), (322, 97, 1), (343, 184, 1), (344, 184, 1), (344, 185, 1), (345, 184, 1), (345, 185, 1), (346, 183, 1), (346, 184, 1), (346, 185, 1), (347, 184, 1), (347, 185, 1), (348, 185, 1), (371, 171, 1), (371, 172, 1), (371, 173, 1), (372, 171, 1), (372, 172, 1), (372, 173, 1), (372, 174, 1), (372, 175, 1), (373, 172, 1), (373, 173, 1), (373, 174, 1), (373, 175, 1), (374, 172, 1), (374, 173, 1), (374, 174, 1), (374, 175, 1), (375, 174, 1), (401, 187, 1), (402, 187, 1), (402, 188, 1), (402, 189, 1), (403, 188, 1), (404, 188, 1), (420, 174, 1), (421, 174, 1), (421, 175, 1), (422, 174, 1), (422, 175, 1), (423, 174, 1), (423, 175, 1), (424, 174, 1), (424, 175, 1), (425, 174, 1), (425, 175, 1), (426, 174, 1), (426, 175, 1), (427, 137, 1), (440, 192, 1), (440, 194, 1), (441, 194, 1), (445, 58, 1), (445, 59, 1)]

for (x, y, scale_found_in) in faces:
    img_2 = cv2.rectangle(img_2, (x, y), (x+window_size[1]*scale_found_in, y+window_size[0]*scale_found_in), (1), 1)
io.imshow(img_2)

io.show()

