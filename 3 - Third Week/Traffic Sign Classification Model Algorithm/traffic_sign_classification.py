"""
Author: Isaac Monroy
Project Title: Traffic Sign Classification Model Algorithm
Description:
    The algorithm's objective is to train a Convolutional Neural 
    Network (CNN) model to recognize and classify traffic signs 
    accurately from a Traffic Sign dataset. The model employs data 
    augmentation to increase the diversity of the training data. 
    After training, the model is saved to a file, and its performance
    is evaluated on a test dataset.
"""
# Import modules
import os  # For handling file paths and directories
from PIL import Image  # To manipulate images for data preprocessing
import numpy as np  # For numerical operations on arrays
import pandas as pd  # To handle the dataset
import matplotlib.pyplot as plt  # For plotting graphs
import cv2  # For image processing
import tensorflow as tf  # To utilize TensorFlow backend
from sklearn.model_selection import train_test_split  # For splitting the data
from sklearn.metrics import accuracy_score  # For accuracy measurement
from keras.utils import to_categorical  # For categorical encoding
from keras.models import Sequential, load_model  # For building and loading models
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout  # For model layers
from keras.preprocessing.image import ImageDataGenerator  # For data augmentation

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create image data generator with augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest"
)

# Intialize data and labels empty lists
data = []
labels = []

# Get current working directory path
cur_path = os.getcwd()

def open_image(images, with_path=True, label=True):
    """ 
    Function for opening and processing images 
    """
    for img in images:
        try:
            if with_path:
                image = Image.open(path + '\\' + img)
            else:
                image = Image.open(img)
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            if label:
                labels.append(traffic_sign)
        except:
            print("Error loading image")
    return data, labels    

# Retrieving the images and their labels 
for traffic_sign in range(43):
    path = os.path.join(cur_path,'train',str(traffic_sign))
    images = os.listdir(path)
    data, labels = open_image(images)

#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), 
                                                    test_size=0.2, random_state=42)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Create model and add the following layers to it
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=43, activation='softmax'))
# Categorial Cross-Entropy loss function and Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Initialize epochs and batch size
epochs = 20
batch_size = 32

# Train the model with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_test, y_test),
    steps_per_epoch=X_train.shape[0] // batch_size,
    verbose=1
)

# Saving the trained model
model.save("my_model.h5")

# Evaluate the model on the test set
_, test_acc = model.evaluate(X_test, y_test, verbose=0)
print('Test accuracy:', test_acc)

y_test = pd.read_csv('Test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

# Open and process test images
data,_ = open_image(imgs, with_path=False, label=False)

X_test=np.array(data)

# Make a prediction on the set of test images
predict_x = model.predict(X_test) 
classes_x = np.argmax(predict_x,axis=1)

#Accuracy with the test data
print(accuracy_score(labels, classes_x))