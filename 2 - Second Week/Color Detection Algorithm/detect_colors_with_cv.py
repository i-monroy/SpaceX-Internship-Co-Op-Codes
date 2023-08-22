"""
Author: Isaac Monroy
Project Title: Color Detection Algorithm
Description:
    Provided with an image that is in the RGB format,
    the algorithm utilizes the OpenCV module's method
    of using a double-click mouse event to identify the
    location and extract the RGB values. With a loaded
    CSV file containing color data, the algorithm
    processes the RGB values to find the closest color
    approximation and ultimately depicts the detected
    color in the form of a rectangle along with the
    name of the color and its RGB values.
"""

# Import necessary modules
import cv2 # Image handling and mouse event detection.
import pandas as pd # Reading and manipulating the CSV file containing color data.

# Initialize the image that shall be used
img_path = 'colorpic.jpg'
org_img = cv2.imread(img_path)

# Load CSV file containing color data and assign column names
column_names=["Color","Color Name","Hex Code","R","G","B"]
colors = pd.read_csv('colors.csv', names=column_names)

# Initialize RGB values and Color Name
B, G, R = 0, 0, 0
color_name = None

# Function to compute the smallest distance from the given RGB values to the color entries in the CSV file
def get_colorname(R, G, B):
    # Compute the smallest distance to find the color
    min_d = min([abs(R - int(c["R"])) + abs(G - int(c["G"])) + abs(B - int(c["B"])) for _, c in colors.iterrows()])
    # Return the color name that corresponds to the smallest distance
    return colors.loc[[abs(R - int(c["R"])) + abs(G - int(c["G"])) + abs(B - int(c["B"])) == min_d for _, c in colors.iterrows()], "Color Name"].iloc[0]

# Function to handle mouse double-click event, extract RGB values and draw the detected color
def draw_function(event_name, x, y, flags, param):
    if event_name == cv2.EVENT_LBUTTONDBLCLK:
        B, G, R = org_img[y, x]
        color_name = get_colorname(R, G, B)
        # Construct a string containing the color name and RGB values
        rgb_and_colorname = f"{color_name} - Red:{R} Green:{G} Blue:{B}"
        # Draw the detected color and the details
        cv2.rectangle(org_img, (25, 25), (700, 75), (int(B), int(G), int(R)), -1)
        cv2.putText(org_img, rgb_and_colorname, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

# Set mouse callback and display the window
cv2.namedWindow('detect colors')
cv2.setMouseCallback('detect colors', draw_function)

# Main loop to keep detecting colors
while True:
    cv2.imshow('detect colors', org_img)
    if cv2.waitKey(1) == 27: # Exit on pressing 'esc' key
        break

cv2.destroyAllWindows()
