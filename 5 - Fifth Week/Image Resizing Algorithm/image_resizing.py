"""
Author: Isaac Monroy
Project Title: Image Resizing and Feature Extraction Algorithm
Description:
    This algorithm performs three main tasks on a given image:
        1. Resizes the image to a specified size (256x256 pixels).
        2. Compresses the image to reduce the memory size.
        3. Utilizes a pre-trained VGG16 model from TensorFlow to extract features
           from the image, storing them in an .npy file.

    The input to the algorithm is the path of an image, which must be in the same
    directory or a valid absolute path. The output is a compressed and resized image
    file, along with a file containing the extracted features.
"""

# Import necessary modules
import os  # For interacting with the operating system
import cv2  # For image processing
import numpy as np  # For numerical operations
from tensorflow.keras.applications import VGG16  # Pre-trained VGG16 model
from tensorflow.keras.preprocessing import image as img_utils  # For image utilities
from tensorflow.keras.applications.vgg16 import preprocess_input  # Preprocessing for VGG16

# Function to resize an image to the target size
def resize_image(input_image, target_size):
    return cv2.resize(input_image, target_size)

# Function to extract features using a pre-trained VGG16 model
def extract_features(resized_image, model):
    x = img_utils.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

# Function to save extracted features to a file
def save_features_to_file(features, file_path):
    np.save(file_path, features)

# Function to load an image from a file
def load_image_from_file(image_path):
    return cv2.imread(image_path)

# Path to the input image
image_path = "path/to/image.jpg"

# Initialize VGG16 model for feature extraction
vgg_model = VGG16(weights='imagenet', include_top=False)

# Check if the image file exists
if os.path.exists(image_path):
    input_image = load_image_from_file(image_path)
    target_size = (256, 256)
    resized_image = resize_image(input_image, target_size)
    cv2.imwrite('./resized_image.jpg', resized_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
    image = cv2.imread('./resized_image.jpg')
    features = extract_features(image, vgg_model)
    save_features_to_file(features, './features.npy')
    cv2.imshow("Resized Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Failed to load input image. Image file does not exist.")
