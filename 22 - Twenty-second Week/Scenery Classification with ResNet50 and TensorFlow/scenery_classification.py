"""
Author: Isaac Monroy
Title: Scenery Classification with ResNet50 and TensorFlow
Description: 
    This script defines and trains a deep learning model for the task
    of classifying scenery images by using the LSUN dataset. The model 
    is built on top of the pre-trained ResNet50 architecture provided 
    by TensorFlow. Data for training, validation, and testing is read 
    from LMDB datasets using a custom data generator class, which allows
    us to efficiently handle large datasets by only loading images into 
    memory in small batches. The training data is further split into
    training and test sets to better evaluate the model's performance.
    The training process includes the use of TensorBoard for monitoring
    the model's performance, as well as metrics such as precision and recall.
    After training, the model is tested on a separate test set, and the 
    results are reported. Finally, the model is saved to the disk and can 
    be reloaded for further usage.
"""
import lmdb # Read the training and validation datasets stored in LMDB format.
import os # Use it to join paths while setting up the directories for the LMDB datasets.
import numpy as np # Used for data generator for calculations like floor division and array creation.
from io import BytesIO # Convert binary image data read from LMDB into a format that PIL's Image can work with.
from PIL import Image # Open images and resize them to the required dimensions for the model.
import tensorflow as tf # Backbone for defining and training the deep learning model.
# Base for our own model and function to prepare input for ResNet50 model.
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
# Used for defining the model architecture, converting labels into a binary matrix representation, and saving/loading the trained model.
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import datetime # Used it for generating timestamps for our TensorBoard logs.
from tensorflow.keras.models import load_model # Used to load a saved model
from sklearn.model_selection import train_test_split # Used to split the training data between train and test

class DataGenerator(tf.keras.utils.Sequence):
    """
    Data generator class for handling large dataset. This class is 
    a helper to process batches of data from LMDBs for training.
    """
    def __init__(self, lmdb_dirs, labels, data_indexes, batch_size, num_classes, shuffle=True):
        """
        Initialization method for data generator class.
        """
        self.lmdb_dirs = lmdb_dirs
        self.labels = labels
        self.data_indexes = data_indexes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.envs = [lmdb.open(lmdb_dir, readonly=True) for lmdb_dir in self.lmdb_dirs]
        self.on_epoch_end()

    def on_epoch_end(self):
        """
        Method executed at the end of each epoch. If shuffle is True, 
        shuffle the indexes.
        """
        self.indexes = self.data_indexes.copy()
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        indices = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(indices)
        return X, y

    def __data_generation(self, indices):
        """
        Generate data containing batch_size samples.
        """
        X = []
        y = []

        for i, key in indices:
            with self.envs[i].begin() as txn:
                image_bin = txn.get(key)
                image = Image.open(BytesIO(image_bin))
                
                # ResNet50 requires the input image size to be 224x224
                image = image.resize((224, 224))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image = preprocess_input(np.array(image))
                X.append(image)
                y.append(to_categorical(self.labels[i], num_classes=self.num_classes))

        return np.array(X), np.array(y)

def split_data(lmdb_dirs, labels, train_ratio=0.8):
    """
    Splits the data into training and test sets.
    80% training 20% test.
    """    
    data = []
    # Loop through LMDB directories and add the index and key to the data list
    for i, lmdb_dir in enumerate(lmdb_dirs):
        env = lmdb.open(lmdb_dir, readonly=True)
        with env.begin() as txn:
            for key, _ in txn.cursor():
                data.append((i, key))
    
    # Shuffle the data to ensure a random distribution
    np.random.shuffle(data)
    # Split the data into training and test datasets according to the given ratio
    train_data, test_data = train_test_split(data, test_size=1-train_ratio)
    return train_data, test_data

# Number of classes to classify
num_classes = 3

# Load and configure the ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False)

# Adding extra layers to our model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # global spatial average pooling layer
x = Dense(1024, activation='relu')(x)  # fully-connected layer

# Output layer -- we have <num_classes> classes
predictions = Dense(num_classes, activation='softmax')(x)

# Final model to train
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer='rmsprop', # RMSprop optimization algorithm
    loss='categorical_crossentropy', # Categorical crossentropy loss function for multi-class classification
    metrics=[
        'accuracy', # Tracking accuracy as a metric
        tf.keras.metrics.Precision(), # Tracking precision as a metric
        tf.keras.metrics.Recall() # Tracking recall as a metric
    ]
)

# Parent folder path for images
folder_path = 'D:/Isaac/4 - Software Automation Engineer/22 - Twenty-second Week'

# Lists for training and validation images for each scenery
lmdb_dirs_train = ([os.path.join(folder_path, "church_outdoor_train_lmdb"), 
                    os.path.join(folder_path, "classroom_train_lmdb"),
                    os.path.join(folder_path, "conference_room_train_lmdb")])

lmdb_dirs_val = ([os.path.join(folder_path, "church_outdoor_val_lmdb"), 
                  os.path.join(folder_path, "classroom_val_lmdb"),
                  os.path.join(folder_path, "conference_room_val_lmdb")])

# 0 for church_outdoor, 1 for classroom, 2 for conference room
labels = [0, 1, 2]

# Split training and testing data
train_data, test_data = split_data(lmdb_dirs_train, labels)

val_data = []
# Loop through validation LMDB directories and append index and key to the validation data list
for i, lmdb_dir in enumerate(lmdb_dirs_val):
    env = lmdb.open(lmdb_dir, readonly=True)
    with env.begin() as txn:
        for key, _ in txn.cursor():
            val_data.append((i, key))

# Setup our data generators
training_generator = DataGenerator(lmdb_dirs_train, labels, train_data, batch_size=32, num_classes=3)
test_generator = DataGenerator(lmdb_dirs_train, labels, test_data, batch_size=32, num_classes=3)
validation_generator = DataGenerator(lmdb_dirs_val, labels, val_data, batch_size=32, num_classes=3)

# Setting up TensorBoard for monitoring
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Training the model
model.fit(training_generator, validation_data=validation_generator, epochs=10, callbacks=[tensorboard_callback])

# Save the trained model
model.save('scene_class_3.h5')  # creates a HDF5 file 'scene_class.h5'

# Load the saved model
model = load_model('scene_class_3.h5')

# Evaluate the model using the test data generator
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print("Test Precision:", test_precision)
print("Test Recall:", test_recall)