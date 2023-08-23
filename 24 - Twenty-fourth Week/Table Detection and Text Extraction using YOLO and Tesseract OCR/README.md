# Table Detection and Text Extraction using YOLO and Tesseract OCR

## Author
Isaac Monroy

## Project Description
This code detects tables within images using a YOLO model and extracts the text within them using Tesseract OCR. The tables are preprocessed through resizing, grayscaling, blurring, thresholding, and inverting to optimize text recognition. The extracted text is then organized into a DataFrame and exported to a CSV file. The project aims to automate the extraction of structured information from visual documents such as invoices.

## Libraries Used
- **Ultralyticsplus (YOLO)**: Object (table) detection and rendering results.
- **PIL**: Image manipulation.
- **Pytesseract**: OCR (text extraction).
- **NumPy**: Numerical operations being performed.
- **OpenCV**: Image preprocessing.
- **Pandas**: Organizing extracted data into DataFrame and CSV.
- **difflib (SequenceMatcher)**: Calculating text similarity.

## How to Run
1. Set the path to the Tesseract executable.
2. Define the `detect_tables` function.
3. Define the `similar` function.
4. Set the path to the invoice image.
5. Run the script to detect tables, process them, and extract the content.
6. Review the detected tables and extracted information printed on the console.
7. Check the CSV file 'invoice_info.csv' for the final extracted data.

## Input and Output
- **Input**: A path to an image (JPEG or PNG) containing tables.
- **Output**: A printed list of detected tables and a CSV file containing the extracted text organized into a DataFrame.
