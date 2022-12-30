from xml_parser import parse_haar_cascade_xml
from constants import PADDING, WINDOW_SIZE, SCALE_FACTOR, STARTING_SCALE, DELTA_SHIFT

from utils import integral_image, integral_of_image_squared, variance
from pipeline.cascade_classifier import Cascade_Classifier
from skimage import io
from skimage.feature import canny
from skimage.filters.edges import sobel_h,sobel_v
from skimage.exposure import histogram
import numpy as np
import cv2
import multiprocessing 

def processWork(x_range_scale_integral_image_cascade_height):
    x_start, x_end = x_range_scale_integral_image_cascade_height[0], x_range_scale_integral_image_cascade_height[1]
    scale = x_range_scale_integral_image_cascade_height[2]
    integral_image, cascade = x_range_scale_integral_image_cascade_height[3], x_range_scale_integral_image_cascade_height[4]
    current_window_size = int(WINDOW_SIZE[0] * scale), int(WINDOW_SIZE[1] * scale)
    
    faces = []
    for x in range(x_start, x_end, int(np.ceil(scale * DELTA_SHIFT))):
        for y in range(0, x_range_scale_integral_image_cascade_height[5] - current_window_size[0] + 1, int(np.ceil(scale * DELTA_SHIFT))):
            # TODO:: we should check the edge threshold before starting

            # window = img[y : y+window_size[0], x : x+window_size[1]]
            # computing the integral window of the current x,y window
            integral_window = integral_image[y : y +  current_window_size[0]+1 , x : x + current_window_size[1]+1]
            # normalizing the integral window mean also to zero
            integral_window = integral_window - np.mean(integral_window)
            # computing the integral window of the current x,y window
            integral_window_squared = x_range_scale_integral_image_cascade_height[6][y : y +  current_window_size[0]+1 , x : x + current_window_size[1]+1]
            # computing the variance. used abs to prevent negative variance issues
            integral_window_variance = np.sqrt(abs(variance(integral_window, integral_window_squared)))
            
            
            # print(integral_window_variance)
            if(cascade.complete_pass(integral_window, integral_window_variance, scale)):
                faces.append((x,y,scale))
    return faces

if __name__ == "__main__":
        
    cascade: Cascade_Classifier = parse_haar_cascade_xml()
    img = io.imread('facedetection/data/images/physics.jpg', as_gray=True)
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

        num_processes = 10
        patch_size = (img_width - current_window_size[1]) // num_processes
        pool = multiprocessing.Pool()
        pool = multiprocessing.Pool(processes=num_processes)
        inputs = []
        if patch_size == 0 :
            num_processes = 1
        for i in range(num_processes):
            start = i*patch_size
            end = start + patch_size -1
            if i == num_processes -1:
                end = (img_width - current_window_size[1] + 1)
            inputs.append( (start, end, scale, integral_image, cascade, img_height, integral_of_image_squared))
        
        outputs_async = pool.map_async(processWork, inputs)
        outputs = outputs_async.get()
        # print("Output: {}".format(outputs))
        for output in outputs:
            faces.extend(output)
        
        # increasing the scale by factor : SCALE_FACTOR
        scale = int(np.ceil(SCALE_FACTOR * scale))


    print(faces)

    for (x, y, scale_found_in) in faces:
        img_2 = cv2.rectangle(img_2, (x, y), (x+window_size[1]*scale_found_in, y+window_size[0]*scale_found_in), (1), 1)
    io.imshow(img_2)

    io.show()

