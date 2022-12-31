import cv2
from skimage import io
import numpy as np
from parser1 import parse_haar_cascade_xml
from maincython import whole_loop
import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
# import sys
# sys.path.insert(1,'F:\\University\\Fall 2022\\image\\Face-Detection-and-Pyramid-Blending\\sift')
from sift.main import sift_wrap
from PyramidBlending.pyramid_blending_main import color_blending

#load the tkinter library
root = tk.Tk()
root.withdraw()
#get path to reference image
reference_image_path =filedialog.askopenfilename(title="Choose reference image")
#get path with image with wrong face
image_to_blend = filedialog.askopenfilename(title="Choose image with faces to blend")

#load the cascade classifier
cascade  = parse_haar_cascade_xml()
#read image to blend
img_gray = io.imread(image_to_blend,as_gray=True)
img = io.imread(image_to_blend)
img_gray = cv2.resize(img_gray,(img_gray.shape[1]//2,img_gray.shape[0]//2))
img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))
#get faces from image to blend
faces = whole_loop(img_gray, cascade)

faces = cv2.groupRectangles(faces, 3, eps=0.1)[0]
#x,y top left
#x_End, y_End bottom right
images = []
number_of_faces = 0
for (x, y, x_end, y_end) in faces:
    number_of_faces+=1
    images.append(img[y-15:y_end + 20, x-15:x_end+20])
    # img = cv2.rectangle(img, (x, y), (x_end, y_end), (1), 1)


fig,axes = plt.subplots(1,number_of_faces)
for index in range(number_of_faces):
    axes[index].imshow(images[index])
    #plot the image together 
plt.show()
var = input("choose face to blend:  ")
while int(var) > number_of_faces:
    print("invalid input")
    var = input("choose face to blend:  ")

face_to_blend = images[int(var)-1]
reference_image = io.imread(reference_image_path)
sifted_image,image_mask = sift_wrap(face_image_rgb=face_to_blend,reference_rgb=reference_image)
io.imsave("sifted_image.jpg",sifted_image)
io.imsave("image_mask.jpg",image_mask)
plt.imshow(sifted_image)
plt.title("sift face")
plt.show()

reference_image = cv2.resize(reference_image,(sifted_image.shape[1],sifted_image.shape[0]))
io.imsave("reference_image.jpg",reference_image)
blended_image = color_blending(sifted_image, reference_image, image_mask,sigma=20)
io.imsave("blended_image.jpg",blended_image)
plt.imshow(blended_image)
plt.title("blended face")
plt.show()

