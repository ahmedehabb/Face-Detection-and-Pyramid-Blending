import numpy as np
from constants import PADDING

def integral_image(img):

    # need to pad the image with zeros from left and up to easier computations
    integral_image_result = np.zeros((img.shape[0]+PADDING, img.shape[1] + PADDING))
    integral_image_result[PADDING:, PADDING:] = img
    return np.cumsum(np.cumsum(integral_image_result, axis=0), axis=1)



# All example sub-windows used for training were variance normalized to minimize the effect of different lighting conditions
# Recall that variance = meanSquared - 1/N * sum(x^squared)


# The mean of a sub-window can be computed using the integral image. 
# The sum of squared pixels is computed using an integral image of the image squared
def integral_of_image_squared(img):
    return integral_image(np.power(img, 2))


# x,y are the beginning of the window relative to whole integral image
def variance(integral_window, integral_window_squared):
    
    # width , height of window
    width, height = integral_window.shape[1], integral_window.shape[0]
    
    # num_pixels in the current window
    # removing the padded part of the window in order to calculate the mean wrongly
    num_pixels = (width-PADDING) * (height-PADDING)
    
    # sum of the pixels in the window
    sum_integral_window = (integral_window[height-1, width-1] + integral_window[0, 0] 
    - integral_window[0, width-1] - integral_window[height-1, 0])
    
    # calculating mean
    mean = sum_integral_window / num_pixels

    # sum of the squared pixels
    sum_integral_window_squared = (integral_window_squared[height-1, width-1] + integral_window_squared[0, 0] 
    - integral_window_squared[0, width-1] - integral_window_squared[height-1, 0])

    # returning variance
    return (sum_integral_window_squared / num_pixels) - pow(mean, 2)


# During scanning the effect of image normalization can be achieved by post-multiplying the feature values rather than 
# pre-multiplying the pixels.


# def fn(x,y, width, height, im):
#     integral_window = im[y : y + height + PADDING , x : x + width + PADDING]
#     print(integral_window.shape)
#     print((width+PADDING,height+PADDING))
#     sum_integral_window = integral_window[height, width] + integral_window[0, 0] - integral_window[0, width] - integral_window[height, 0]
#     print(sum_integral_window)

# very important
# img = np.ones((3,3))
# im = integral_image(img)
# width, height = 2,2
# print(im)
# will be larger by row and col due to PADDING
# print(integral_window)
# fn(0,0,2,2,im)



# testing windowing and integral images

# img = np.ones((3,3))
# window_size = 2,2
# integral_window_size = window_size[0]+PADDING ,window_size[1]+PADDING

# im = integral_image(img)
# integral_squared = integral_of_image_squared(img)
# print(im)

# # print(img)
# for x in range(0 , 4-window_size[0] + 1 ):
#     for y in range(0 , 4-window_size[1] + 1 ):
#         window = img[y : y+window_size[0], x : x+window_size[1]]
#         integral_window = im[y : y +  integral_window_size[0] , x : x + integral_window_size[1] ]
#         integral_window_squared = integral_squared[y : y +  integral_window_size[0] , x : x + integral_window_size[1] ]
#         integral_window_variance = variance(integral_window, integral_window_squared)
        
#         # print(integral_window)
#         print(integral_window_variance)
#         # exit()
# y = 1
# x = 1
# window = img[y : y+window_size[0], x : x+window_size[1]]
# integral_window = im[y : y +  integral_window_size[0] , x : x + integral_window_size[1] ]
# integral_squared_window = integral_squared[y : y +  integral_window_size[0] , x : x + integral_window_size[1] ]

# now integral image is my whole world 0,0 -> 25,25
# x, y, rect_width, rect_height = 0,0,2,2

# rect_window = integral_window[y: y + rect_height + PADDING , x : x + rect_width+ PADDING]
# x_end ,y_end  = rect_width + PADDING - 1, rect_height + PADDING - 1
# value = (rect_window[y_end, x_end] + rect_window[0, 0] - rect_window[0, x_end] - rect_window[y_end, 0])


# width, height = integral_window.shape[1], integral_window.shape[0]
# print(width, height)


# num_pixels = (width-PADDING) * (height-PADDING)

# sum_integral_window = (integral_window[height-1, width-1] + integral_window[0, 0] 
#     - integral_window[0, width-1] - integral_window[height-1, 0])
# print(integral_squared, integral_squared_window)

# mean = sum_integral_window / num_pixels

# # sum of the squared pixels
# sum_integral_window_squared = (integral_squared_window[height-1, width-1] + integral_squared_window[0, 0] 
# - integral_squared_window[0, width-1] - integral_squared_window[height-1, 0])

# # returning variance
# var= pow(mean, 2) - (sum_integral_window_squared / num_pixels)

# print( mean , sum_integral_window_squared, var)