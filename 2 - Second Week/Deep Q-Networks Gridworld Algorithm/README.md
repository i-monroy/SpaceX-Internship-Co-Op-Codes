# Deep Q-Networks Gridworld Algorithm

## Author
Isaac Monroy

## Project Description
This algorithm trains a model to play and win the Gridworld game using Deep Q-Networks. The player aims to reach the goal in the least amount of moves without hitting any obstacles. The model learns through experiences, generating Q-values to guide the player towards victory.

## Libraries Used
- **Gridworld**: Custom library to handle the game's environment.
- **numpy**: Used for numerical operations and state manipulation.
- **torch**: The deep learning library used to create the neural network.
- **torch.nn**: Used to define the network's layers and the loss function.
- **torch.optim**: Contains the Adam optimizer used to update the network's weights.
- **random**: Utilized for stochastic elements in the training process.
- **IPython.display**: Used to clear output when printing loss during training.
- **copy**: To create a deep copy of the model for the target network.
- **collections.deque**: To handle the experience replay memory.

## How to Run
1. Ensure all required libraries are installed.
2. Import the `Gridworld` class, or replace it with a compatible class representing the game environment.
3. Run the entire script to train the model and test it on 1000 games, printing the win percentage.

## Input and Output
- **Input**: The model takes the current state of the Gridworld game, represented as a 64-element array.
- **Output**: The model outputs an action to be taken by the player. The final script prints the number of games won and the win percentage over 1000 games.
