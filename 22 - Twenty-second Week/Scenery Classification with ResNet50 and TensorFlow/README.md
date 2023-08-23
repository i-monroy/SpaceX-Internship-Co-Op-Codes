# Scenery Classification with ResNet50 and TensorFlow

## Author
Isaac Monroy

## Project Description
This script defines and trains a deep learning model for classifying scenery images using the LSUN dataset. It builds on the pre-trained ResNet50 architecture by TensorFlow, uses custom data generator class for efficient handling of large datasets, and employs TensorBoard for performance monitoring. The script also includes the training process and evaluation using precision and recall metrics, with the ability to save and reload the model.

## Libraries Used
- **lmdb**: Read the training and validation datasets stored in LMDB format.
- **os**: Utilized for joining paths while setting up the directories for the LMDB datasets.
- **numpy**: Used for calculations like floor division and array creation in data generation.
- **PIL**: Open and resize images to the required dimensions for the model.
- **tensorflow**: Backbone for defining and training the deep learning model, including functions to prepare input for ResNet50 model and additional functions for defining the model architecture.
- **datetime**: Used for generating timestamps for TensorBoard logs.
- **sklearn**: Used for splitting the training data between train and test.

## How to Run
1. Set up your LMDB directories for training and validation datasets.
2. Configure the number of classes to classify and adjust the layers as needed.
3. Compile the model.
4. Set up the parent folder path for images and your LMDB directories and labels.
5. Split training and testing data.
6. Set up data generators for training, testing, and validation.
7. Train the model by running `model.fit`, and optionally, monitor with TensorBoard.
8. Save the trained model using `model.save`.
9. Load the saved model and evaluate using the test data generator.

## Input and Output
- **Input**: The code takes in scenery images from LMDB datasets and processes them using the ResNet50 model.
- **Output**: Outputs include the trained model saved to disk, as well as printed test loss, accuracy, precision, and recall metrics. The training process can also be monitored through TensorBoard.
