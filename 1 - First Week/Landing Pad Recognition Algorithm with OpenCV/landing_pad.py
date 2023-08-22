"""
Author: Isaac Monroy
Project Title: Landing Pad Recognition Algorithm
Description:
    Given an image containing a SpaceX "X" logo or SpaceX landing pad,
    the algorithm recognizes all possible corners from the image and
    classifies whether the image contains a landing pad or not.
"""

# Importing the necessary libraries
import cv2 # OpenCV for image processing
import numpy as np # NumPy for numerical operations
import imutils # imutils for image resizing

# Read the selected image
img_path = 'path/to/landing_pad_input.jpg'
big_img = cv2.imread(img_path)
cv2.imshow('org img', big_img)
cv2.waitKey(0)

# Resize the image to a consistent height
ratio = big_img.shape[0] / 500.0
org = big_img.copy()
img = imutils.resize(big_img, height=500)
cv2.imshow('resizing', img)
cv2.waitKey(0)

# Convert the image to grayscale and apply edge detection
gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
edged_img = cv2.Canny(blur_img, 75, 200)
cv2.imshow('edged', edged_img)
cv2.waitKey(0)

# Find and analyze contours in the image
cnts,_ = cv2.findContours(edged_img.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:20]

# First choice, if the image is just a logo, then this method shall 
# be used to prove whether the image is or not a landing pad

# Iterate through all of the contours and select only 
# the relevant points and store the coordinates.
fin_result = []
duplicates = []
for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    for x in range(len(approx)):
        matrix1 = np.array(approx[x])
        for y in range(len(approx)):
            if y <= len(approx)-1:
                matrix2 = np.array(approx[y])
                matrix_comp = matrix1 == matrix2
                matrix_value_dup = np.isin(matrix2, duplicates, invert=True)
                matrix_dup = np.isin(matrix2, fin_result, invert=True)
                if not matrix_comp.all():
                    if matrix_value_dup.all():
                        result = matrix1 - matrix2
                        if(result[0][0] < 10 and result[0][0] > -10):
                            duplicates.append(approx[y])
                        elif matrix_dup.all():
                            fin_result.append(matrix2)

# Classify the image based on the number of
# found corners and denote whether the image
# is a landing pad or not.
if len(fin_result) >= 9 and len(fin_result) <= 13:
    classify_1 = True
else:
    classify_1 = False

# Second choice, if the image has more objects around it, detect and
# calculate for the centroid of a circular shape that is found within 
# a drone ship landing pad.

# Iterate through the detected contours and see whether the
# desired number of corners can be found along with a specific
# range for the centroid.
fin_result = []
find_once = 0
for c in cnts:
    doc = None
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx) == 4 and find_once == 1:
        fin_result = approx
        find_once += 1
    elif len(approx) == 8 and find_once == 0:
        fin_result = approx
        M = cv2.moments(approx)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        find_once += 1 
    elif len(approx) == 4 and find_once == 0:
        fin_result = approx
        find_once += 1
        M = cv2.moments(approx)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        find_once += 1 
range_cx = cx >= 350 and cx <= 450
range_cy = cy >= 200 and cy <= 300

# Classify the image based on the calculated
# centroid and denote whether the image is a
# landing pad or not.
if range_cx and range_cy:
    classify_2 = True
else:
    classify_2 = False

# Display the final result
if classify_1 == True or classify_2 == True:
    classify_3 = 'Landing Pad Detected'
else:
    classify_3 = 'No Landing Pad Found'
cv2.putText(img, classify_3, (0, 490), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
    