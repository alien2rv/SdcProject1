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
from moviepy.editor import VideoFileClip
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#reading in an image
def perform_magic_py(img):
    image = mpimg.imread(img)
    img=os.path.basename(img)
    #printing out some stats and plotting
    #print('This image is:', type(image), 'with dimensions:', image.shape)
    #grayscale image
    mod_image=processing.grayscale(image)
    #gaussian blur
    mod_image=processing.gaussian_blur(img=mod_image,kernel_size=5)
    mod_image2=processing.color_select(threshold=180,image=mod_image)
    #canny edge detection
    mod_image=processing.canny(img=mod_image,low_threshold=50, high_threshold=150)
    #masking unnecessary edges using triangular masking  
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(450, 290), (490, 290), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges= processing.region_of_interest(img=mod_image, vertices=vertices)
    mod_image2= processing.region_of_interest(img=mod_image2, vertices=vertices)
    #hough transform
    mod_image=processing.hough_lines(img=masked_edges, rho=1, theta=np.pi/180, threshold=int(10), min_line_len=int(150), max_line_gap=int(150))
    #interopolate hough transformed image to main image
    mod_image = processing.weighted_img(mod_image,image)
    a = np.zeros_like(mod_image2)
    mod_image2=np.dstack((mod_image2,a,a))
    mod_image2 = processing.weighted_img(mod_image2,image)
    plt.imshow(mod_image)
    plt.title("lanelines_"+img)
    plt.figure()
    paths=os.path.join("test_images","results","lanelines",img)
    plt.imsave(paths,mod_image)
    plt.imshow(mod_image2)
    plt.title("linesegments_"+img)
    plt.figure()
    paths=os.path.join("test_images","results","linesegments",img)
    plt.imsave(paths,mod_image2)

def process_image(image):
    #image = mpimg.imread(img)
    #print('This image is:', type(image), 'with dimensions:', image.shape)
    mod_image=processing.grayscale(image)
    plt.imshow(mod_image)
    mod_image=processing.gaussian_blur(img=mod_image,kernel_size=5)
    #mod_image=processing.color_select(threshold=100,image=mod_image)
    mod_image=processing.canny(img=mod_image,low_threshold=50, high_threshold=150)
    vertices=np.array([[(0,mod_image.shape[0]),(450, 290), (490, 290), (mod_image.shape[1],mod_image.shape[0])]], dtype=np.int32)
    masked_edges= processing.region_of_interest(img=mod_image, vertices=vertices)
    #print(masked_edges.shape)
    mod_image=processing.hough_lines(img=masked_edges, rho=2, theta=np.pi/180, threshold=int(35), min_line_len=int(150), max_line_gap=int(100))
    mod_image = processing.weighted_img(mod_image,image)    
    return mod_image
    
# folder= 'test_images'
# for filename in os.listdir(folder):
#     if ".jpg" in filename:
#         print(f"processing file : {filename}")
#         perform_magic_py(os.path.join(folder,filename))
#         #print(os.path.join(folder,filename))

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
video_folder = 'test_videos'
video_output_folder = 'test_videos_output'
for filename in os.listdir(video_folder):
    #clip1 = VideoFileClip("test_videos/solidYellowLeft.mp4").subclip(0,5)
    print(filename)
    clip1 = VideoFileClip(os.path.join(video_folder,filename))
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.write_videofile(os.path.join(video_output_folder,filename), audio=False)
