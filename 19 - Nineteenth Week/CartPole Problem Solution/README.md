# CartPole Problem Solution

## Author
Isaac Monroy

## Project Description
This program applies a Deep Q-Learning algorithm to solve the CartPole balancing problem from the OpenAI Gym. The algorithm uses a multi-layered neural network to approximate the Q-value function. The network is trained using experiences stored in a replay buffer to minimize correlations in observations. To balance exploration and exploitation, we use an epsilon-greedy policy. Furthermore, the model's weights are updated periodically to a target model to provide more stable learning targets. The code also includes functionality for normalizing the rewards using a StandardScaler from the sklearn library.

## Libraries Used
- **gym**: OpenAI gym for creating the CartPole environment.
- **numpy**: Used for numerical operations.
- **tensorflow**: Used for creating and training the neural network.
- **keras**: Used for creating the model structure.
- **random**: Used for epsilon-greedy action selection and sampling from the replay buffer.
- **sklearn.preprocessing.StandardScaler**: Used for reward normalization.

## How to Run
1. Install the required libraries (gym, numpy, tensorflow, keras, random, sklearn).
2. Create the CartPole environment using OpenAI gym.
3. Define and initialize the ReplayBuffer class to store experience replay.
4. Define the neural network model using Keras.
5. Train the model using the experiences stored in the ReplayBuffer, periodically update the target model, and apply epsilon-greedy policy.
6. Run the episodes, and the model will learn to balance the CartPole.

## Input and Output
- **Input**: The initial state of the CartPole environment.
- **Output**: Action to take to balance the CartPole, based on the Q-value approximation. The code also prints the reward for each episode.
