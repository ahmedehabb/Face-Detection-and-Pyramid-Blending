import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import generateDescriptors,removeDuplicateKeypoints,convertKeypointsToInputImageSize,localizeExtremumViaQuadraticFit,computeKeypointsWithOrientations

def number_of_octaves(image):
       return int(round(np.log(min(image.shape)) / np.log(2) - 1))



def generate_base_image(image,sigma,assumed_blur):
    #we will upsample the image
    image = cv2.resize(image,(0,0),fx=2,fy=2,interpolation=cv2.INTER_LINEAR)
    #we added the 0.01 to prevent zeros
    differnece_of_sigmas = np.sqrt(max((sigma**2) - (2*assumed_blur)**2,0.01 ))
    
    return cv2.GaussianBlur(image,(0,0),sigmaX=differnece_of_sigmas,sigmaY=differnece_of_sigmas)

#Step 1: generate scale space for image
# def generate_scale_space(image,number_of_octaves,number_of_blurring):
#     #for each octave level, we will generate for the same image size gaussian blurred number_of_blurring times
    
    
def generate_sigma_kernels(sigma,num_of_interval):
    # number_of_images_per_octave = num_of_interval + 3 #as shown in paper
    # k = 2**(1./number_of_images_per_octave) #as shown in paper
    # gaussian_kernels = np.zeros(number_of_images_per_octave)
    # gaussian_kernels[0] = sigma
    # for index in range(1,number_of_images_per_octave):
    #     gaussian_kernels[index] = k * gaussian_kernels[index-1] #as shown in lecture slides
        
        
    num_images_per_octave = num_of_interval + 3
    k = 2 ** (1. / num_of_interval)
    gaussian_kernels = np.zeros(num_images_per_octave)  # scale of gaussian blur necessary to go from one blur scale to the next within an octave
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generate_gaussian_images(image,number_of_octaves,gaussian_kernels):
    gaussian_images = []
    for octave_index in range(number_of_octaves):
        gaussian_images_in_octave = []
        
        #TODO if failed do as the paper
        for kernel in gaussian_kernels[0:]:
            image = cv2.GaussianBlur(image,(0,0),sigmaX=kernel,sigmaY=kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        
        #TODO: check why we choose the middle image as our base
        octave_base = gaussian_images_in_octave[-3] 
        #down sample the image to be half the size
        image = cv2.resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=cv2.INTER_NEAREST)
    return np.array(gaussian_images,dtype=object)
            

def generate_difference_of_gaussian_images(gaussian_images):
    
    number_of_octaves = len(gaussian_images)
    number_of_images_per_octave = len(gaussian_images[0])
    DOG_images = []
    for each_octave in gaussian_images:
        current_DOG_images= []
        for index in range(0,number_of_images_per_octave-1):
            current_DOG_images.append(cv2.subtract(each_octave[index+1],each_octave[index]))
        DOG_images.append(current_DOG_images)
    return DOG_images
            

            
def built_in_functions(gaussian_images,i,j,image_index,octave_index,num_intervals,dog_images_in_octave,sigma,contrast_threshold,image_border_width):
    localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index, num_intervals, dog_images_in_octave, sigma, contrast_threshold, image_border_width)
    if localization_result is not None:
        keypoint, localized_image_index = localization_result
        keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gaussian_images[octave_index][localized_image_index])
        return keypoints_with_orientations
    return None
    
def is_pixel_extrma(first_window,second_window,third_window,threshold):
    """
    If i am bigger than all 26 pixel or iam smaller than smaller
    printz

    """
    # print("first window size",first_window,first_window.shape)
    center_pixel = second_window[1,1] #pick the middle pixel
    # print("center pixel is",center_pixel)
    if(abs(center_pixel) > threshold): #to avoid edges
        if center_pixel > 0:
            
           return np.all(center_pixel >=first_window) and \
                    np.all(center_pixel >= third_window) and \
                    np.all(center_pixel >= second_window[0, :]) and \
                    np.all(center_pixel >= second_window[2, :]) and \
                    (center_pixel >= second_window[1, 0]) and \
                    (center_pixel >= second_window[1, 2])
        elif center_pixel < 0:
            return np.all(center_pixel <= first_window)and \
                   np.all(center_pixel <= third_window)and \
                   np.all(center_pixel <= second_window[0, :])and \
                   np.all(center_pixel <= second_window[2, :])and \
                   center_pixel <= second_window[1, 0] and \
                   center_pixel <= second_window[1, 2]
    return False
    

def get_key_points(gaussian_images,DOG_images,image_border_width,sigma,contrast_threshold=0.03,number_of_intervals=5):
    threshold = math.floor(0.5 * contrast_threshold / number_of_intervals * 255)
    keypoints = []
    
    for octave_index,dog_image_in_octave in enumerate(DOG_images):
        #loop through each octave and corresponding DOG images
        for image_index, (first_image,second_image,third_image) in enumerate(zip(dog_image_in_octave,dog_image_in_octave[1:],dog_image_in_octave[2:])):
            for i in range(image_border_width, first_image.shape[0] - image_border_width):
                for j in range(image_border_width, first_image.shape[1] - image_border_width):
                    if is_pixel_extrma(first_image[i-1:i+2, j-1:j+2], second_image[i-1:i+2, j-1:j+2], third_image[i-1:i+2, j-1:j+2], threshold):
                        # we have current_octave, image_index_in that octave,#i,j key point of that image
                        #now we will use built in functions to 
                        key_points_with_orientations = built_in_functions(gaussian_images,i,j,image_index,octave_index,2,dog_image_in_octave,sigma,contrast_threshold,image_border_width)
                        if key_points_with_orientations is not None:
                            for keypoint_with_orientation in key_points_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointsToInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)
    return keypoints,descriptors


def wrapper_SIFT(img):
    img = img.astype('float32')
    sigma=1.6
    assumed_blur=0.5
    number_of_gaussian_kernels=2 #as in slides
    base_image = generate_base_image(img,sigma, assumed_blur)
    num_of_octaves= number_of_octaves(base_image)
    gaussian_kernels = generate_sigma_kernels(sigma,number_of_gaussian_kernels)
    gaussian_images = generate_gaussian_images(base_image,num_of_octaves,gaussian_kernels)
    dog_images = generate_difference_of_gaussian_images(gaussian_images)
    key_points,descriptors = get_key_points(gaussian_images=gaussian_images,DOG_images=dog_images,sigma=sigma,image_border_width=5,contrast_threshold=0.03,number_of_intervals=2)
    return key_points,descriptors
    