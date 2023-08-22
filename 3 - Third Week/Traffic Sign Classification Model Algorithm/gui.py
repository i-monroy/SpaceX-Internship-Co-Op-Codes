"""
Author: Isaac Monroy
Project Title: Traffic Sign Classification GUI
Description:
    The GUI's objective is to use a pre-trained 
    traffic sign classification model to predict 
    the traffic signs from the images that users 
    upload. After the user uploads an image, the 
    algorithm displays the predicted traffic sign 
    along with its meaning on the window.
"""
# Import modules
import tkinter as tk  # For building the GUI interface
from tkinter import filedialog  # For file dialog in the GUI
from tkinter import *  # For various GUI components
from PIL import ImageTk, Image  # To handle and display images in the GUI
import os  # For handling file paths
import numpy as np  # For numerical operations on arrays
from keras.models import load_model  # To load the trained model for classification

# Load the pre-trained model to classify traffic signs
model = load_model('my_model.h5')

# Dictionary containing the mapping of class labels to traffic sign names
classes = { 1:'Speed limit (20km/h)',
            2:'Speed limit (30km/h)',      
            3:'Speed limit (50km/h)',       
            4:'Speed limit (60km/h)',      
            5:'Speed limit (70km/h)',    
            6:'Speed limit (80km/h)',      
            7:'End of speed limit (80km/h)',     
            8:'Speed limit (100km/h)',    
            9:'Speed limit (120km/h)',     
           10:'No passing',   
           11:'No passing veh over 3.5 tons',     
           12:'Right-of-way at intersection',     
           13:'Priority road',    
           14:'Yield',     
           15:'Stop',       
           16:'No vehicles',       
           17:'Veh > 3.5 tons prohibited',       
           18:'No entry',       
           19:'General caution',     
           20:'Dangerous curve left',      
           21:'Dangerous curve right',   
           22:'Double curve',      
           23:'Bumpy road',     
           24:'Slippery road',       
           25:'Road narrows on the right',  
           26:'Road work',    
           27:'Traffic signals',      
           28:'Pedestrians',     
           29:'Children crossing',     
           30:'Bicycles crossing',       
           31:'Beware of ice/snow',
           32:'Wild animals crossing',      
           33:'End speed + passing limits',      
           34:'Turn right ahead',     
           35:'Turn left ahead',       
           36:'Ahead only',      
           37:'Go straight or right',      
           38:'Go straight or left',      
           39:'Keep right',     
           40:'Keep left',      
           41:'Roundabout mandatory',     
           42:'End of no passing',      
           43:'End no passing veh > 3.5 tons' }

# Create a Tkinter window
root = Tk()

# Set the window size
root.geometry("800x600")
root.title('Traffic Sign Classification')

# Open image file and resize it to fit window
image = Image.open("city_road.jpg")
image = image.resize((800, 600), Image.Resampling.LANCZOS)

# Convert the image to a Tkinter-compatible photo image
photo = ImageTk.PhotoImage(image)

def upload_and_classify():
    """ 
    Upload a traffic sign image and classify it 
    """
    try:
        # Upload image and resize it to show it successfully
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((upload_image_label.winfo_width()*20),(upload_image_label.winfo_height()*20)))
        im = ImageTk.PhotoImage(uploaded)

        # Create a label widget for upload image
        upload_image_label.configure(image=im)
        upload_image_label.image = im

        # Reset the heading label text
        heading_label.configure(text='Traffic Sign Classification')
        
        # Predict what the image can be and display it
        # on the window
        image = Image.open(file_path)
        image = image.resize((30,30))
        image = np.expand_dims(image, axis=0)
        image = np.array(image)
        pred = model.predict([image])[0]
        classes_pred=np.argmax(pred)
        sign = classes[classes_pred+1]
        print(sign)
        prediction_label.configure(text=sign)

    except:
        pass

# Create a label with the image
label = Label(root, image=photo)
label.pack()

# Create heading, upload image and precition labels
heading_label = Label(root, text="Traffic Sign Classification", font=("Arial", 20), bg="white", fg="black")
heading_label.place(relx=0.5, rely=0.15, anchor=CENTER)
upload_image_label = Label(root, bg="white", image=None)
upload_image_label.place(relx=0.5, rely=0.5, anchor=CENTER)
prediction_label = Label(root, bg="white", font=("Arial", 16), fg="black")
prediction_label.place(relx=0.5, rely=0.35, anchor=CENTER)

# Create button with a command to execute upload and classify function
button = Button(root, text="Upload sign image and classify it", font=("Arial", 12), command=upload_and_classify)
button.place(relx=0.5, rely=0.8, anchor=CENTER)

# Run the Tkinter event loop
root.mainloop()

