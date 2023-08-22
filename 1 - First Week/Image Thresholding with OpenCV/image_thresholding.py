"""
Author: Isaac Monroy
Project Title: Image Thresholding Algorithm
Description: 
    This script demonstrates the application of different thresholding 
    techniques on an input image. It reads an image, applies five types 
    of thresholding (Binary, Binary Inverted, To Zero, To Zero Inverted, 
    Truncation), and displays the original and thresholded images using 
    matplotlib.
"""
import cv2 as cv # Image processing and thresholding techniques
import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For displaying the images of the threshold types

# Set the threshold values for classifying the pixel values
# and set maximum pixel value.
classify_pix_val = 127
max_pix_val = 255

# Read path and load image
image_path = 'path/to/image.png'
image = cv.imread(image_path, 0)

# List of threshold types
threshold_types = [cv.THRESH_BINARY, cv.THRESH_BINARY_INV, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV, cv.THRESH_TRUNC]

# Apply simple thresholding with the following types:
#  - Binary, Binary Inv, To Zero, To Zero Inv, and Trunc
images = [image] + [cv.threshold(image, classify_pix_val, max_pix_val, thresh_type)[1] for thresh_type in threshold_types]

# List of image titles
titles = ['Original image', 'Binary', 'Binary Inv', 'To Zero', 'To Zero Inv', 'Trunc']

# Loop through both lists and create a plot representation of the six images, and show
# the images obtained.
for i, (img, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(img, 'gray', vmin=0, vmax=255)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
plt.show()
