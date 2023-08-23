"""
Author: Isaac Monroy
Title: Convolutional Neural Networks For Urban Sound Classification
Description: 
    This script uses Convolutional Neural Networks (CNNs) for classifying
    urban sounds. The dataset used is UrbanSound8K which is a collection of
    8732 labeled sound excerpts of urban sounds from 10 classes (air conditioner
    , car horn, children playing and more). The sound files are preprocessed 
    by extracting Mel Frequency Cepstral Coefficients (MFCCs) from them. The 
    extracted features are then used to train a CNN model. Based on the results,
    the training data is used to plot a loss per epochs plot and accuracy per 
    epochs to depict how well the model did during training. At last, the performance
    of the model is evaluated using a confusion matrix and a classification report.
"""

# Import necessary libraries
import os  # Interaction with the filesystem
import numpy as np  # For numerical computations
import pandas as pd  # Data manipulation and analysis
import librosa as lb  # For music and audio analysis
import IPython.display as ipd
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import tensorflow as tf  # For machine learning and numerical computations
from tensorflow import keras  # High-level API to build and train models in TensorFlow
from tensorflow.keras import Sequential, layers  # For creating the Sequential model and adding layers to it
from tensorflow.keras.utils import to_categorical  # Converting class vector to binary class matrix
from sklearn.model_selection import train_test_split  # Splitting the dataset into training and testing sets
from sklearn.metrics import confusion_matrix, classification_report  # To evaluate the performance of the classification
from tqdm import tqdm

# Load data and display metadata
metadata = pd.read_csv('path/to/UrbanSound8K.csv') # Replace with your path to csv file
classes = metadata.groupby('classID')['class'].unique()

def extract_features(path):
    """
    Extract Mel-frequency cepstral coefficients (MFCCs) from a sound file.
    """
    # Load the audio file
    data, sample_rate = lb.load(path)
    # Compute MFCCs and then its mean
    mfccs = lb.feature.mfcc(data, n_mfcc=128)
    mfccs_mean = np.mean(mfccs, axis=1) 
    return mfccs_mean

# Extracting features for each audio file
features, labels = [], []
for index, row in tqdm(metadata.iterrows()):
    # Replace with path to audio
    audio_path = 'path/to/sound_datasets/urbansound8k/audio/' + 'fold' + str(row['fold']) + '/' + str(row['slice_file_name'])
    features.append(extract_features(audio_path))
    labels.append(row['classID'])

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

# One-hot encoding
labels = to_categorical(labels)

# Split data into train, validation and test sets
features_train_val, features_test, labels_train_val, labels_test = train_test_split(features, labels, test_size=0.1, stratify=labels, random_state=387)
features_train, features_valid, labels_train, labels_valid = train_test_split(features_train_val, labels_train_val, test_size=0.2, stratify=labels_train_val, random_state=387)

# Reshape the data to be 3D
features_train = np.reshape(features_train, (features_train.shape[0], features_train.shape[1], 1))
features_valid = np.reshape(features_valid, (features_valid.shape[0], features_valid.shape[1], 1))

def create_model():
    """
    Create the Convolutional Neural Network model for sound classification.
    """
    model = Sequential()

    # First Conv1D layer
    model.add(layers.Conv1D(64, 3, padding='same', input_shape=(128, 1)))  # Convolutional layer
    model.add(layers.Activation('relu'))  # ReLU activation
    model.add(layers.MaxPooling1D(pool_size=2))  # Max pooling

    # Second Conv1D layer
    model.add(layers.Conv1D(128, 3, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # Flattening the output from Conv layers to feed it into a dense layer
    model.add(layers.Flatten())

    # Dense Layer 1
    model.add(layers.Dense(1024))  # Fully connected layer
    model.add(layers.Activation('relu'))  # ReLU activation
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

    # Dense Layer 2
    model.add(layers.Dense(512))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(10))
    model.add(layers.Activation('softmax'))  # Softmax activation for multi-class classification

    return model

# Create and compile the model
model = create_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(features_train, labels_train, validation_data=(features_valid, labels_valid), epochs=30)

# Convert training history to dataframe
history_df = pd.DataFrame(history.history)

# Plotting the loss per epochs
plt.figure(figsize=(20,8))
plt.plot(history_df[['loss','val_loss']])
plt.legend(['loss','val_loss'])
plt.title('Loss Per Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# Plotting the accuracy per epochs
plt.figure(figsize=(20,8))
plt.plot(history_df[['accuracy','val_accuracy']])
plt.legend(['accuracy','val_accuracy'])
plt.title('Accuracy Per Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Reshape the data to be 3D
features_test = np.reshape(features_test, (features_test.shape[0], features_test.shape[1], 1))

# Predict the labels for the test set and convert from one-hot encoded vectors to class numbers
labels_true = np.argmax(labels_test, axis=1)
labels_pred = np.argmax(model.predict(features_test), axis=1)

# Print confusion matrix and classification report
print('\nConfusion Matrix :\n\n')
print(confusion_matrix(labels_true, labels_pred))
print('\n\nClassification Report : \n\n', classification_report(labels_true, labels_pred))