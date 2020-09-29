# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 22:43:35 2020

@author: Ravi
"""

#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from processing import processing
import os
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
#grayscale image
mod_image=processing.grayscale(image)
#gaussian blur
mod_image=processing.gaussian_blur(img=mod_image,kernel_size=5)
#canny edge detection
mod_image=processing.canny(img=mod_image,low_threshold=100, high_threshold=200)
#masking unnecessary edges using triangular masking
left_bottom = [100, 535]
right_bottom = [900, 535]
apex = [500, 250]
#mask = np.zeros_like(mod_image)   
#ignore_mask_color = 255  
#imshape = image.shape
vertices = np.array([left_bottom, right_bottom, apex], dtype=np.int32)
#cv2.fillPoly(mask, vertices, ignore_mask_color)
#masked_edges = cv2.bitwise_and(mod_image, mask)
masked_edges= processing.region_of_interest(img=mod_image, vertices=vertices)
#hough transform
mod_image=processing.hough_lines(img=masked_edges, rho=1, theta=np.pi/180, threshold=50, min_line_len=300, max_line_gap=150)
#print('This image is:', type(grey), 'with dimensions:', grey.shape)
#interopolate hough transformed image to main image
mod_image = processing.weighted_img(mod_image,image)
plt.imshow(mod_image)
plt.imsave("mod_image_"+"solidWhiteRight.jpg",mod_image)

folder= 'test_images'
for filename in os.listdir(folder):
    print(filename)