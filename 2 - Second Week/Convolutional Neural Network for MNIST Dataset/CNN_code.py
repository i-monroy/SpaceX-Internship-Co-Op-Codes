"""
Author: Isaac Monroy
Project Title: Convolutional Neural Network for MNIST Dataset
Description:
    This code defines, trains, and evaluates a convolutional neural 
    network (CNN) for recognizing digits from the MNIST dataset. The 
    CNN is trained on the training data, and the model's accuracy is 
    evaluated on the testing data. Various optimizers like Adagrad, 
    RMSprop, and Adam are used to experiment with the training process, 
    and the results are plotted to visualize the performance.
"""

# Import necessary modules
import torch # Used for defining and training the neural network model.
import torch.nn as nn # Used for building neural network layers and defining loss functions.
from torch.autograd import Variable # Allows automatic differentiation of tensors.
from torchvision.datasets import MNIST # Dataset class for the MNIST dataset.
from torchvision.transforms import ToTensor # Transformation to convert image data to tensor format.
from torch.utils.data import DataLoader # Helps in loading and batching data efficiently.
from torch import optim # Contains optimization algorithms (SGD, Adagrad, RMSprop, etc.).
import matplotlib.pyplot as plt # Used for plotting graphs and visualizing data.
import numpy as np # For numerical operations.
import copy # Used for deep copying objects such as model state dictionaries.

# Set a random seed for reproducibility
torch.manual_seed(0)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, in_dim, feature_dim, out_dim):
        super (Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim),
            nn.ReLU()
            )

    def forward(self, inputs):
        return self.classifier(inputs)

# Load MNIST dataset
train_set = MNIST('.', train=True, download=True, transform=ToTensor())
test_set = MNIST('.', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# Model parameters
IN_DIM, FEATURE_DIM, OUT_DIM = 784, 256, 10
model = Net(IN_DIM, FEATURE_DIM, OUT_DIM)

# Loss function and optimizers
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
momentum = 0.9
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Train model for the number of epochs selected and
# print the current epoch and loss of the model
model.train()
epochs = 40
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        b_imgs = Variable(images)
        b_labels = Variable(labels)       
        output = model(b_imgs)
        loss = loss_fn(output, b_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch + 1, epochs, i + 1, len(train_loader), loss.item()))

# Save the created model
torch.save(model.state_dict(), 'mnist.pt')

# Load the model and assign the next values for layers
# Nodes: input layer=784, hidden layer=256, output layer=10
IN_DIM, FEATURE_DIM, OUT_DIM = 784, 256, 10
model = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
model.load_state_dict(torch.load('mnist.pt'))

# Copy dictionary with its values and assign it to
# the following variable
opt_state_dict = copy.deepcopy(model.state_dict())

# Print the architecture of the network
for param_tensor in opt_state_dict:
    print(param_tensor, "\t",
         opt_state_dict[param_tensor].size())

# Create new model with randomly initialized 
# parameters
model_rand = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
rand_state_dict = copy.deepcopy(model_rand.state_dict())

# Create new model test model
test_model = Net(IN_DIM, FEATURE_DIM, OUT_DIM)
test_state_dict = copy.deepcopy(test_model.state_dict())

# Compute the average loss over the loaded data set
def inference(testloader, model, loss_fn):
    avg_loss = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            avg_loss += loss
    avg_loss /= len(testloader)
    return avg_loss

# Create linear interpolation by computing alpha and beta
results = []
for alpha in torch.arange(-2, 2, 0.05):
    beta = 1.0 - alpha
    for p in opt_state_dict:
        test_state_dict[p] = (opt_state_dict[p] * beta +
                            rand_state_dict[p] * alpha)

    # Load the resulted test model, and store the
    # obtained loss
    model.load_state_dict(test_state_dict)
    loss = inference(train_loader, model, loss_fn)
    results.append(loss.item())

# Plot the incurred error over the alpha parameter
plt.plot(np.arange(-2, 2, 0.05), results, 'ro')
plt.ylabel('Incurred Error')
plt.xlabel('Alpha')

# Initialize the momentum and randomly chosen steps
rand_walk = [torch.randint(-10, 10, (1,1)) for x in range(100)]
momentum = 0.1
momentum_rand_walk = [torch.randint(-10, 10, (1,1)) for x in range(100)]

# Iterate over the chosen steps and record each new step
for i in range(1, len(rand_walk) - 1):
    prev = momentum_rand_walk[i-1]
    rand_choice = torch.randint(-10, 10, (1,1)).item()
    new_step = momentum * prev + (1 - momentum) * rand_choice
    momentum_rand_walk[i] = new_step

# Delete previous graph and plot the new steps with momentum
plt.clf()
plt.plot(momentum_rand_walk[:-1])
plt.xlabel('Steps')

# Instantiate Conv2d model
model = nn.Conv2d(1,32,3)

# Adagrad optimizer
optimizer = optim.Adagrad(model.parameters(),
    lr = 1e-2,
    weight_decay = 0,
    initial_accumulator_value = 0)
print(optimizer)

# RMSprop optimizer
optimizer = optim.RMSprop(model.parameters(),
    lr = 1e-2,
    alpha = 0.99,
    eps = 1e-8,
    weight_decay = 0,
    momentum = 0)
print(optimizer)

# Adam optimizer
optimizer = optim.Adam(model.parameters(),
    lr = 1e-3,
    betas = (0.9, 0.999),
    eps = 1e-08,
    weight_decay = 0,
    amsgrad = False)
print(optimizer)
