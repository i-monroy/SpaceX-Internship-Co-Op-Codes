# Image Resizing and Feature Extraction Algorithm

## Author
Isaac Monroy

## Project Description
This algorithm resizes a given image to a specified size of 256x256 pixels, compresses it to reduce the memory size, and then utilizes a pre-trained VGG16 model from TensorFlow to extract features from the image. The features are then stored in an .npy file.

## Libraries Used
- **os:** For interacting with the operating system.
- **cv2:** For image processing, such as resizing and loading images.
- **numpy:** For numerical operations, including working with arrays.
- **tensorflow.keras.applications:** For accessing the pre-trained VGG16 model.
- **tensorflow.keras.preprocessing:** For image utilities like conversion to array.

## How to Run
1. Ensure the required libraries are installed.
2. Provide a valid image path within the code.
3. Run the script, and the resized image and features file will be generated in the specified location.

## Input and Output
- **Input:** Path of an image in the same directory or a valid absolute path.
- **Output:** A compressed and resized image file of size 256x256 pixels, along with a file containing the extracted features (.npy file).
