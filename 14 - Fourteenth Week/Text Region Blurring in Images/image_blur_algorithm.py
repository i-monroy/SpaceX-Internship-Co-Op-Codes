"""
Author: Isaac Monroy
Title: Text Region Blurring in Images
Description:
    This script loads a trained EAST (Efficient and Accurate Scene Text) deep
    learning model to detect regions of text in an image, and applies a blurring 
    effect to these regions. It supports processing single images and batches of 
    images.

    The algorithm first loads a trained EAST deep learning model, then reads an 
    image (or images), and preprocesses it by resizing it to a size the model 
    expects. The script creates a blob from the image which is used as the input 
    to the model.

    The script uses the model to make predictions on the image, extracting scores 
    and geometry information. It then uses this information to calculate the 
    bounding boxes for regions in the image where text was detected. Non-Maxima 
    Suppression is used to handle overlapping bounding boxes.

    Each detected text region is then blurred using a Gaussian blur and the original image 
    is updated with these blurred regions. This effectively obscures the text in the image. 
    The script saves the blurred image to a specified directory. 

    Note: Blurring percentage and other parameters can be modified within the script as needed.
"""
# File I/O libraries
import os # Required for directory and file interactions
import sys # Provide access to variables used or maintained by the Python interpreter

# Image processing and machine learning libraries
import numpy as np # Required for numerical operations
import cv2 # OpenCV library used for image processing tasks
from imutils.object_detection import non_max_suppression # Required for handling overlapping bounding boxes

def load_east_model(model_path):
    """
    Function to load the EAST text detection model 
    from the provided path.
    """
    try:
        return cv2.dnn.readNet(model_path)
    except Exception as e:
        print(f"Failed to load the model. Error: {str(e)}")
        sys.exit(1)

def preprocess_image(image):
    """
    Function to preprocess the image to fit the EAST 
    text detector model input.
    """
    try:
        # Convert grayscale image to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Get original image dimensions
        (H, W) = image.shape[:2]
        # Set new image dimensions
        (newW, newH) = (320, 320)

        # Compute ratio for restoring image dimensions
        rW = W / float(newW)
        rH = H / float(newH)
        
        # Resize image
        image = cv2.resize(image, (newW, newH))
        
        return image, (rW, rH)
    except Exception as e:
        print(f"Failed to preprocess the image. Error: {str(e)}")
        sys.exit(1)

def create_blob(image):
    """
    Function to create a blob from the image to use as
    the EAST text detector input.
    """
    try:
        (H, W) = image.shape[:2]
        return cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    except Exception as e:
        print(f"Failed to create blob from image. Error: {str(e)}")
        sys.exit(1)

def decode_predictions(scores, geometry, confidence_threshold):
    """
    Function to decode the predictions into the bounding
    box coordinates that will determine the blurred areas
    of the image.
    """
    try:
        # Prepare empty lists for rects and confidences
        rects = []
        confidences = []
        
        # Get dimensions from the scores
        (numRows, numCols) = scores.shape[2:4]
        
        # For each cell in the output map of the EAST text detector
        for y in range(0, numRows):
            # Extract data for bounding box computation
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            
            # For each cell in the output map of the EAST text detector
            for x in range(0, numCols):
                # Skip low confidence detections
                if scoresData[x] < confidence_threshold:
                    continue
                
                # Compute the offset factor for the current cell
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                
                # Extract the rotation angle for the prediction
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                # Compute the dimensions of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                
                # Compute the starting and ending coordinates for the bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                
                # Add the bounding box coordinates and score to our lists
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])
                
        return rects, confidences
    except Exception as e:
        print(f"Failed to decode predictions. Error: {str(e)}")
        sys.exit(1)

def blur_boxes(image, boxes, rW, rH):
    """
    Function to apply a Gaussian blur over the regions of 
    the image that were detected by the EAST text detector.
    """
    try:
        expansion_factor = 0.06
        for (startX, startY, endX, endY) in boxes:
            # Scale bounding box coordinates based on the respective ratios
            startX = int(startX * rW * (1 - expansion_factor))
            startY = int(startY * rH * (1 - expansion_factor))
            endX = int(endX * rW * (1 + expansion_factor))
            endY = int(endY * rH * (1 + expansion_factor))

            # Ensure the bounding box coordinates do not exceed the dimensions of the image
            height, width = image.shape[:2]
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(width - 1, endX), min(height - 1, endY)
            
            # Extract the region of interest
            roi = image[startY:endY, startX:endX]
            # Apply Gaussian blur on the region of interest
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            # Replace the original region of interest with the blurred one
            image[startY:endY, startX:endX] = blurred_roi
        return image
    except Exception as e:
        print(f"Failed to blur boxes. Error: {str(e)}")
        sys.exit(1)

def east_detect(image_path, model_path):
    """
    Function to use the EAST text detector to detect text in 
    the image, and then applies a Gaussian blur on the detected 
    regions.
    """
    # Load EAST model
    net = load_east_model(model_path)
    try:
        # Read the image
        image = cv2.imread(image_path)
    except Exception as e:
        print(f"Failed to read the image. Error: {str(e)}")
        sys.exit(1)
    orig = image.copy()
    
    # Preprocess the image and create a blob from the preprocessed image
    image, (rW, rH) = preprocess_image(image)
    blob = create_blob(image)

    # Define the two output layer names for the EAST detector model
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    try:
        # Forward pass the blob through the model
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
    except Exception as e:
        print(f"Failed to make forward pass. Error: {str(e)}")
        sys.exit(1)

    # Decode the predictions into bounding boxes and apply non-maxima suppression
    confidence_threshold = 0.7
    rects, confidences = decode_predictions(scores, geometry, confidence_threshold)
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    # Blur the boxes in the original image
    blurred_image = blur_boxes(orig, boxes, rW, rH)
    
    return blurred_image

def process_batch(image_paths, model_path):
    """
    Function to process a batch of images by detecting text and 
    blurring it.
    """
    # Directory for saving the processed images
    processed_imgs_path = "./blurred_images"
    if not os.path.exists(processed_imgs_path):
        os.mkdir(processed_imgs_path)
    
    # Process each image
    for image_path in image_paths:
        try:
            # Detect text in the image and blur it
            blurred_image = east_detect(image_path, model_path)
            # Save the processed image
            base_name = os.path.basename(os.path.splitext(image_path)[0])
            output_path = base_name + "_output" + os.path.splitext(image_path)[1]
            cv2.imwrite(os.path.join(processed_imgs_path, output_path), blurred_image)
            print(f'Successfully processed and saved image: {os.path.join(processed_imgs_path, output_path)}')
        except Exception as e:
            print(f'An error occurred while processing image: {image_path}. Error: {e}')

# Directory with input images
imgs_folder_path = "./images_folder"
# Get list of image paths
image_paths = [os.path.join(imgs_folder_path, img) for img in os.listdir(imgs_folder_path) if img.lower().endswith((".png", ".jpg", ".jpeg"))]

# Path to the EAST text detection model
model_path = "./model/frozen_east_text_detection.pb"
# Process all images
process_batch(image_paths, model_path)