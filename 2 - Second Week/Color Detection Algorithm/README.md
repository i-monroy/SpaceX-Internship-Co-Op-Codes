# Color Detection Algorithm

## Author:
Isaac Monroy

## Project Description
Provided with an image in the RGB format, this algorithm utilizes OpenCV to detect the color at a location double-clicked by the user. It extracts the RGB values at the clicked location and matches them to a pre-defined CSV file containing color data. The closest matching color is then depicted on the screen in the form of a rectangle, along with the color's name and its RGB values.

## Libraries Used
- **cv2 (OpenCV):** For image handling and mouse event detection.
- **pandas:** For reading and manipulating the CSV file containing color data.

## How to Run
1. Install the required libraries (cv2, pandas).
2. Ensure the image ('colorpic.jpg') and CSV file ('colors.csv') are in the same directory as the script.
3. Run the script.
4. Double-click on the displayed image to detect colors.
5. Press the 'esc' key to exit the program.

## Input and Output
- **Input:** An image in RGB format and a CSV file containing color data.
- **Output:** A continuous display of the image with detected colors being depicted as rectangles along with their names and RGB values.
