# Convolutional Neural Networks For Urban Sound Classification

## Author
Isaac Monroy

## Project Description
This script uses Convolutional Neural Networks (CNNs) for classifying urban sounds. The dataset used is UrbanSound8K which is a collection of 8732 labeled sound excerpts of urban sounds from 10 classes (air conditioner, car horn, children playing, and more). The sound files are preprocessed by extracting Mel Frequency Cepstral Coefficients (MFCCs) from them. The extracted features are then used to train a CNN model. Based on the results, the training data is used to plot a loss per epochs plot and accuracy per epochs to depict how well the model did during training. At last, the performance of the model is evaluated using a confusion matrix and a classification report.  

## Libraries Used
- **os**: Interaction with the filesystem
- **numpy**: Numerical computations
- **pandas**: Data manipulation and analysis
- **librosa**: Music and audio analysis
- **IPython**: Display utilities
- **matplotlib**: Creating static, animated, and interactive visualizations
- **tensorflow**: Machine learning and numerical computations
- **keras**: High-level API to build and train models in TensorFlow
- **scikit-learn**: Data preprocessing and evaluation

## How to Run
1. Make sure all libraries are installed.
2. Download the UrbanSound8K dataset.
3. Update the paths for the CSV file and audio files in the code.
4. Run the script to train the CNN model, plot the graphs, and evaluate the model using the confusion matrix and classification report.

## Input and Output
**Input**: Audio files from the UrbanSound8K dataset, and associated metadata in a CSV file.  
**Output**: A trained CNN model that classifies urban sounds into 10 classes, along with the loss and accuracy plots for the training process, and a confusion matrix and classification report for evaluating the model's performance.
