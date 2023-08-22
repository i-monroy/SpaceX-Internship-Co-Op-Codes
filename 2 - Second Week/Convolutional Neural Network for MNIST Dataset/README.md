# Convolutional Neural Network for MNIST Dataset

## Author
Isaac Monroy

## Project Description
This project involves the creation and training of a Convolutional Neural Network (CNN) to recognize and accurately evaluate the images from the MNIST dataset. The CNN is trained to predict the number that corresponds to the images in the dataset, aiming to accurately predict any image from this dataset.

### Libraries Used
- **torch:** Used for defining and training the neural network model.
- **torch.nn:** Used for building neural network layers and defining loss functions.
- **torch.autograd.Variable:** Allows automatic differentiation of tensors.
- **torchvision.datasets:** Dataset class for the MNIST dataset.
- **torchvision.transforms:** Transformation to convert image data to tensor format.
- **torch.utils.data:** Helps in loading and batching data efficiently.
- **torch.optim:** Contains optimization algorithms (SGD, Adagrad, RMSprop, etc.).
- **matplotlib.pyplot:** Used for plotting graphs and visualizing data.
- **numpy:** For numerical operations.
- **copy:** Used for deep copying objects such as model state dictionaries.

## How to Run
1. Ensure that the required libraries are installed.
2. Download the MNIST dataset by running the code (it will automatically download if not present).
3. Set the desired number of epochs and other hyperparameters.
4. Run the code to train the model on the MNIST dataset.

## Input and Output
- **Input:** The MNIST dataset, divided into training and testing sets.
- **Output:** A trained CNN model saved to 'mnist.pt' capable of predicting the numerical value corresponding to an input image from the MNIST dataset. Various plots can also be generated to visualize the performance and characteristics of the model.
