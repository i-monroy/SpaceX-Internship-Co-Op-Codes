# Twin Delayed Deep Deterministic (TD3) Policy for BipedalWalker

## Author 
Isaac Monroy

## Project Description
This script implements the Twin Delayed Deep Deterministic (TD3) algorithm for training a BipedalWalker agent. TD3 is a reinforcement learning algorithm that is a variant of Deep Deterministic Policy Gradients (DDPG) with additional tricks to deal with problems like function approximation error. It uses two Q-functions to reduce overestimation bias and delayed policy updates to reduce variance. The agent uses an actor-critic architecture with the PyTorch library for defining and training the networks. The OpenAI gym's BipedalWalker-v3 environment is used as a training ground for the agent.

## Libraries Used
- **torch**: Used for creating and training the neural networks used in the TD3 algorithm.
- **torch.nn**: Used for creating the actor and critic networks.
- **torch.nn.functional**: Used for implementing activation functions.
- **torch.optim**: Used for defining the optimizer that updates the parameters of the neural networks.
- **numpy**: Used for numerical operations like generating random numbers for exploration.
- **gym**: Used to create the BipedalWalker-v3 environment.

## How to Run
1. Install the required libraries (PyTorch, NumPy, gym).
2. Define the TD3 class, Actor and Critic network classes, and replay buffer as provided in the code.
3. Set hyperparameters such as learning rate, gamma, batch size, etc.
4. Run the training loop to train the agent using the TD3 algorithm.
5. (Optional) Monitor the training progress by logging rewards or visualizing the agent's performance.

## Input and Output
- **Input**: The code takes the state of the BipedalWalker environment and uses the TD3 policy to select an action.
- **Output**: The selected action is executed in the environment, and the resulting next state, reward, and done flag are processed. The final output includes the trained TD3 policy and performance metrics such as rewards per episode.
