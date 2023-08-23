# Text Region Blurring in Images

## Author 
Isaac Monroy

## Project Description
This project loads a pre-trained EAST (Efficient and Accurate Scene Text) deep learning model to detect regions of text in an image or batch of images, then blurs these regions. It's useful for obscuring sensitive text information in images.

## Libraries Used
- **os**: For directory and file operations
- **sys**: To access variables used by the Python interpreter
- **numpy**: For numerical operations
- **cv2**: OpenCV library for image processing
- **imutils**: For handling overlapping bounding boxes

## How to Run
1. Place images in the specified folder (default: ./images_folder)
2. Ensure the EAST model file is in the specified location (default: ./model/frozen_east_text_detection.pb)
3. Run the script

## Input and Output
- **Input**: Path to a directory containing images (JPG, JPEG, PNG)
- **Output**: Processed images with blurred text regions, saved to a specified directory (default: ./blurred_images)
