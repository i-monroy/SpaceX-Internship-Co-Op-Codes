# Proximal Policy Optimization for Lunar Lander

## Author
Isaac Monroy

## Project Description
This script implements the Proximal Policy Optimization (PPO) algorithm to train a Lunar Lander agent. PPO is a type of policy gradient method for reinforcement learning, which uses a surrogate objective to improve the policy. The agent uses two separate networks, an actor and a critic, to estimate the policy and value function, respectively. The script uses PyTorch to define and train the networks, and the OpenAI gym's LunarLander environment to train the agent.

## Libraries Used
- **torch**: Used to create and train the neural networks in the PPO algorithm.
- **torch.nn**: Used for creating the actor and critic networks.
- **torch.optim**: Used for defining the optimizer that updates the parameters of the neural networks.
- **gym**: Used to create the Lunar Lander environment.
- **torch.distributions.Categorical**: Used for creating a probability distribution over the possible actions.

## How to Run
1. Ensure you have the required libraries installed (torch, gym).
2. Save the code in a file with a `.py` extension, such as `ppo_lunar_lander.py`.
3. Run the script using the command `python ppo_lunar_lander.py`.
4. The script will train the agent on the LunarLander-v2 environment for a defined number of episodes or until a solved reward is achieved.
5. The trained model will be saved as a `.pth` file when the training is solved.

## Input and Output
- **Input**: The code takes the LunarLander-v2 environment as an input to train the agent.
- **Output**: The code outputs the training progress, including the average length and reward per episode, and saves the trained policy in a `.pth` file once the environment is solved.
