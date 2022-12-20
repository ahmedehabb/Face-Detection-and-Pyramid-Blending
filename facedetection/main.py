from parser import parse_haar_cascade_xml
from constants import PADDING, WINDOW_SIZE, SCALE_FACTOR, STARTING_SCALE, DELTA_SHIFT

from utils import integral_image, integral_of_image_squared, variance
from pipeline.cascade_classifier import Cascade_Classifier
from skimage import io
from skimage.feature import canny
from skimage.filters.edges import sobel_h,sobel_v
from skimage.exposure import histogram
import numpy as np
import cv2

cascade: Cascade_Classifier = parse_haar_cascade_xml()
img = io.imread('facedetection/data/images/company.jpeg', as_gray=True)
# img = cv2.resize(img, (640,480))
img_2 = np.copy(img)

img_width, img_height = img.shape[1], img.shape[0]

# normalizing image mean and variance
mean = np.mean(img)
img = img - mean
# # All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions.
img = img / np.sqrt(np.var(img))
# print(np.mean(img), np.var(img))

# TODO:: GET EDGE IMAGE AND PUT CORRECT THRESHOLD
#  WILL THRESHOLD DEPEND ON WINDOW SIZE? 
# I THINK YES SINCE EDGES INCREASE AS SIZE OF WINDOW INCREASE 

# edges = canny(img, sigma = 2)
# # calculating integral image for edges so we can check it before testing windows
# edge_strength_integral = integral_image(edges)

# io.imshow(edges > 0.2)
# io.show()
# exit()

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
            # TODO:: we should check the edge threshold before starting

            # window = img[y : y+window_size[0], x : x+window_size[1]]
            # computing the integral window of the current x,y window
            integral_window = integral_image[y : y +  integral_window_size[0] , x : x + integral_window_size[1]]
            # normalizing the integral window mean also to zero
            integral_window = integral_window - np.mean(integral_window)
            # computing the integral window of the current x,y window
            integral_window_squared = integral_of_image_squared[y : y +  integral_window_size[0] , x : x + integral_window_size[1]]
            # computing the variance. used abs to prevent negative variance issues
            integral_window_variance = np.sqrt(abs(variance(integral_window, integral_window_squared)))
            
            
            # print(integral_window_variance)
            if(cascade.complete_pass(integral_window, integral_window_variance, scale)):
                print("complete", integral_window_variance)
                faces.append((x,y,scale))
            
    
    # increasing the scale by factor : SCALE_FACTOR
    scale = int(np.ceil(SCALE_FACTOR * scale))


print(faces)

for (x, y, scale_found_in) in faces:
    img_2 = cv2.rectangle(img_2, (x, y), (x+window_size[1]*scale_found_in, y+window_size[0]*scale_found_in), (1), 1)
io.imshow(img_2)

io.show()

