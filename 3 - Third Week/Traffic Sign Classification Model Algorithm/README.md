# Traffic Sign Classification Project

## Author
Isaac Monroy

## Project Description
A two-part project aimed at classifying traffic signs. The first part is training a model using a Sequential CNN and data augmentation to recognize and classify traffic sign images. The second part involves implementing a GUI that utilizes the trained model to predict and display the traffic signs from user-uploaded images.

## Libraries Used:
#### Code 1:
- `os`: For handling file paths and directories.
- `PIL.Image`: To manipulate images for data preprocessing.
- `numpy`: For numerical operations on arrays.
- `pandas`: To handle the dataset.
- `matplotlib.pyplot`: For plotting graphs.
- `cv2`: For image processing.
- `tensorflow`: To utilize TensorFlow backend.
- `sklearn.model_selection`, `sklearn.metrics`: For model training and evaluation.
- `keras`: For building, training, and saving the model.

#### Code 2:
- `tkinter`: For building the GUI interface.
- `PIL.ImageTk`, `PIL.Image`: To handle and display images in the GUI.
- `os`: For handling file paths.
- `numpy`: For numerical operations on arrays.
- `keras.models`: To load the trained model for classification.

## How to Run:
1. Ensure that all the required libraries are installed.
2. Train the model using the code in Code 1 and save it as 'my_model.h5'.
3. Run the GUI code in Code 2.
4. Upload an image of a traffic sign in the GUI.
5. View the predicted traffic sign and its description.

## Input and Output:
- **Input**:
  - Code 1: Images of traffic signs.
  - Code 2: User-uploaded image of a traffic sign.
- **Output**:
  - Code 1: Trained model file.
  - Code 2: Predicted traffic sign and its description displayed in the GUI.
