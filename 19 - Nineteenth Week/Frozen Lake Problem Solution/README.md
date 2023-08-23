# Frozen Lake Problem Solution

## Author: Isaac Monroy

## Project Description
This program applies a Deep Q-Learning algorithm to solve the FrozenLake problem from the OpenAI Gym. The algorithm uses a three-layered neural network to approximate the Q-value function. The network is trained using experiences stored in a replay buffer to minimize correlations in observations. To balance exploration and exploitation, we use an epsilon-greedy policy. The code also includes functionality for saving and loading the model.

## Libraries Used
- **gym**: OpenAI gym is used for the environment.
- **numpy**: Used for numerical operations.
- **tensorflow**: Utilized as the deep learning framework.
- **random**: Used for epsilon-greedy action selection and minibatch selection.
- **collections**: Used for the Deque data structure for the replay buffer.

## How to Run
1. Install the required libraries using pip or conda.
2. Save the code in a file with a .py extension.
3. Open a terminal or command prompt.
4. Navigate to the directory where the file is saved.
5. Run the command `python filename.py`, replacing "filename" with the name of your file.

## Input and Output
- **Input**: The code takes parameters such as number of episodes, learning rate, discount rate, and exploration rate as inputs for the Q-learning algorithm.
- **Output**: The output includes the score for each episode, epsilon value, and loss if applicable. Model weights can be saved to disk.
