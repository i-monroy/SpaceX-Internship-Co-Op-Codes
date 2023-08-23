"""
Author: Isaac Monroy
Title: Table Detection and Text Extraction using YOLO and Tesseract OCR
Description: 
    This code detects tables within images using a YOLO model and 
    extracts the text within them using Tesseract OCR. The tables 
    are preprocessed through resizing, grayscaling, blurring, 
    thresholding, and inverting to optimize text recognition. The 
    extracted text is then organized into a DataFrame and exported 
    to a CSV file. The project aims to automate the extraction of 
    structured information from visual documents such as invoices.
"""
from ultralyticsplus import YOLO, render_result # Object (table) detection and rendering results
from PIL import Image # Image manipulation
import pytesseract # OCR (text extraction)
import numpy as np # Numerical operations being performed
import cv2 # Image preprocessing
import pandas as pd # Organizing extracted data into DataFrame and CSV
from difflib import SequenceMatcher # Calculating text similarity

# Set path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'path\to\tesseract.exe'

def detect_tables(image_path):
    """
    Detects tables in the given image using a YOLO model.
    """
    # Load pre-trained model
    model = YOLO('foduucom/table-detection-and-extraction')
    
    # Set model parameters
    model.overrides['conf'] = 0.25 # Confidence threshold for Non-Maximum Suppression (NMS)
    model.overrides['iou'] = 0.45  # Intersection-over-Union (IoU) threshold for NMS
    model.overrides['agnostic_nms'] = False  # Whether NMS is class-agnostic
    model.overrides['max_det'] = 10 # Maximum number of detections per image
    
    # Perform inference on the image
    results = model.predict(image_path)
    
    # Extract bounding boxes from the results
    bounding_boxes = results[0].boxes.xyxy
    
    # Render and show the result
    render = render_result(model=model, image=image_path, result=results[0])
    render.show()
    
    return bounding_boxes.tolist()  # Convert bounding_boxes to a list

def similar(a, b):
    """
    Computes the similarity ratio between two strings.
    """
    return SequenceMatcher(None, a, b).ratio()

# For an image, set its path and detect its tables
image = 'path/to/jpg/or/png/picture'
tables = detect_tables(image)
print("Detected tables:", tables)

# Load the image with PIL
image = Image.open(image)

# Iterate over detected tables and process them
for box in tables:
    crop_box = (box[0], box[1], box[2], box[3])
    cropped_table = image.crop(crop_box)
    
    # Convert the cropped table from PIL format to OpenCV format
    cropped_table_cv = np.array(cropped_table)
    cropped_table_cv = cv2.cvtColor(cropped_table_cv, cv2.COLOR_RGB2BGR)

    # Apply preprocessing steps: resizing, grayscaling, blurring, thresholding, and inverting
    cropped_table_cv = cv2.resize(cropped_table_cv, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(cropped_table_cv, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    roi = cv2.bitwise_not(thresh)

    # Extract text from the preprocessed image using Tesseract
    extracted_text = pytesseract.image_to_string(roi, config='--oem 3')

# Split the Tesseract output into lines and prepare dictionaries for the DataFrame
lines = extracted_text.split('\n')
data = {"Qty": [], "Description": [], "Unit Price": [], "Net Amount": []}
current_column = None
keywords = ["Qty", "Description", "Unit Price", "Net Amount"]

# Iterate over lines and parse the content
for line in lines:
    line = line.strip()
    
    # Skip empty lines
    if not line:
        continue

    found_keyword = False

    # Check if the line is similar to any of the keywords or contains part of them
    for keyword in keywords:
        if similar(line, keyword) > 0.6 or any(part in line for part in keyword.split()):
            current_column = keyword
            found_keyword = True
            break

    # If the line is not a keyword and not empty, add it to the DataFrame
    if not found_keyword and current_column:
        data[current_column].append(line)

# Balance the columns by filling with empty strings
max_len = max(len(column) for column in data.values())
for column in data:
    data[column] += [''] * (max_len - len(data[column]))

# Create a DataFrame and save to CSV
df = pd.DataFrame(data)
print(df)
df.to_csv('invoice_info.csv', index=False)