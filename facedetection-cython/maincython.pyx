import numpy as np
cimport numpy as np
from skimage.feature import canny

from pipeline cimport cascade_classifier

# padding used in integral image
cdef int PADDING = 1
# the scale by which scale variable starts by
cdef int STARTING_SCALE = 1
# factor by which window scale increase each time
cdef float SCALE_FACTOR = 1.25
cdef int DELTA_SHIFT = 1
# defining the window size specs as in constants
cdef int window_size_height = 24
cdef int window_size_width = 24

cdef np.ndarray[np.float64_t, ndim=2] integral_image(np.ndarray[np.float64_t, ndim=2] img):

    # need to pad the image with zeros from left and up to easier computations
    cdef np.ndarray[np.float64_t, ndim=2] integral_image_result = np.zeros((img.shape[0]+PADDING, img.shape[1] + PADDING), dtype=np.float64)
    integral_image_result[PADDING:, PADDING:] = img
    return np.cumsum(np.cumsum(integral_image_result, axis=0), axis=1)


# The mean of a sub-window can be computed using the integral image. 
# The sum of squared pixels is computed using an integral image of the image squared
cdef np.ndarray[np.float64_t, ndim=2] integral_of_image_squared(np.ndarray[np.float64_t, ndim=2] img):
    return integral_image(np.power(img, 2))


# x,y are the beginning of the window relative to whole integral image
cdef float variance(np.ndarray[np.float64_t, ndim=2] integral_window, np.ndarray[np.float64_t, ndim=2] integral_window_squared):
    
    # width , height of window
    cdef int width, height 
    width, height = integral_window.shape[1], integral_window.shape[0]
    
    # num_pixels in the current window
    # removing the padded part of the window in order to calculate the mean of the real data not with padded row and col
    cdef int num_pixels = (width-PADDING) * (height-PADDING)
    
    # sum of the pixels in the window
    cdef float sum_integral_window = (integral_window[height-1, width-1] + integral_window[0, 0] 
    - integral_window[0, width-1] - integral_window[height-1, 0])
    
    # calculating mean
    cdef float mean = sum_integral_window / num_pixels

    # sum of the squared pixels
    cdef float sum_integral_window_squared = (integral_window_squared[height-1, width-1] + integral_window_squared[0, 0] 
    - integral_window_squared[0, width-1] - integral_window_squared[height-1, 0])

    # returning variance
    return (sum_integral_window_squared / num_pixels) - pow(mean, 2)


cpdef list whole_loop(np.ndarray[np.float64_t, ndim=2] img, cascade_classifier.Cascade_Classifier cascade):

    cdef int img_height, img_width
    img_width, img_height = img.shape[1], img.shape[0]

    # normalizing image mean and variance
    cdef float mean = np.mean(img)
    cdef float var = np.sqrt(np.var(img))
    img = img - mean
    # # All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions.
    if(var != 0):
        img = img / var
    # print(np.mean(img), np.var(img))

    # TODO:: GET EDGE IMAGE AND PUT CORRECT THRESHOLD
    #  WILL THRESHOLD DEPEND ON WINDOW SIZE? 
    # I THINK YES SINCE EDGES INCREASE AS SIZE OF WINDOW INCREASE 

    cdef np.ndarray[np.float64_t, ndim=2] img_canny = canny(img, sigma=3).astype(np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] canny_integral_image = integral_image(img_canny)
    # # calculating integral image for edges so we can check it before testing windows
    # edge_strength_integral = integral_image(edges)


    # getting the integral image and integral imag squared

    cdef np.ndarray[np.float64_t, ndim=2] integral_image_ = integral_image(img)
    cdef np.ndarray[np.float64_t, ndim=2] integral_image_squared = integral_of_image_squared(img)


    cdef list faces = []

    cdef int scale = 2
    cdef int max_scale = 5
    cdef int x,y
    cdef float integral_window_variance
    cdef np.ndarray[np.float64_t, ndim=2] integral_window, integral_window_squared
    cdef np.ndarray[np.float64_t, ndim=2] window_canny
    cdef int current_window_size_height, current_window_size_width
    cdef int integral_window_size_height, integral_window_size_width

    while scale < max_scale:
        print(scale)
        # define current scale and window size accordingly
        current_window_size_height = window_size_height * scale
        current_window_size_width = window_size_width * scale

        # defining integral window size to be window size + padding added to correctly index integral window
        integral_window_size_height = current_window_size_height + PADDING
        integral_window_size_width = current_window_size_width + PADDING

        canny_threshold = window_size_height*scale*scale

        for x in range(0, img_width - current_window_size_width + 1, int(np.ceil(scale * DELTA_SHIFT))):
            for y in range(0, img_height - current_window_size_height + 1, int(np.ceil(scale * DELTA_SHIFT))):
                # TODO:: we should check the edge threshold before starting
                window_canny = canny_integral_image[y : y + integral_window_size_height,x : x + integral_window_size_width]
                y1, x1 = 0, 0
                y2, x2 = window_canny.shape[0] - 1, window_canny.shape[1] - 1
                
                total_canny = window_canny[y2, x2] + window_canny[y1, x1] - window_canny[y2, x1] - window_canny[y1, x2]

                if (total_canny < canny_threshold):
                    continue

                # computing the integral window of the current x,y window
                integral_window = integral_image_[y : y +  integral_window_size_height , x : x + integral_window_size_width]
                # normalizing the integral window mean also to zero
                integral_window = integral_window - np.mean(integral_window)
                # computing the integral window of the current x,y window
                integral_window_squared = integral_image_squared[y : y +  integral_window_size_height , x : x + integral_window_size_width]
                # computing the variance. used abs to prevent negative variance issues
                integral_window_variance = variance(integral_window, integral_window_squared)
                if(integral_window_variance > 0):
                    integral_window_variance = np.sqrt(integral_window_variance)
                else:
                    integral_window_variance = 1
                
                
                if(cascade.complete_pass(integral_window, integral_window_variance, scale)):
                    print("complete", integral_window_variance)
                    faces.append((x,y,x+scale*window_size_width, y+scale*window_size_height))
                
        
        # increasing the scale by factor : SCALE_FACTOR
        scale = int(np.ceil(SCALE_FACTOR * scale))


    return faces

