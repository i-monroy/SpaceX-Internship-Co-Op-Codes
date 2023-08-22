"""
Author: Isaac Monroy
Project Title: Context Bandit MDP Algorithm
Description:
    Given a set of 10 slot machines (arms), the goal of 
    the agent is to find the slot machine that yields the 
    highest reward. The algorithm leverages the Markov 
    Decision Process and trains a Neural Network model 
    to achieve the best outcome.
"""

# Import necessary modules
import numpy as np # For numerical operations
import torch # Used for defining and training the neural network model.
import torch.nn as nn # Used for building neural network layers and defining loss functions.
from torch.nn import Sequential # Used for creating a sequential neural network model
import random # Used for generating random numbers
import matplotlib.pyplot as plt # Used for plotting the mean rewards over time to visualize the performance

class ContextBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()
    
    # Number of states equals the number of arms
    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms,arms)
    
    # Get reward based on a probability
    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward
    
    # When selecting an arm, a reward is
    # returned and the state is updated.
    def get_state(self):
        return self.state
    
    def update_state(self):
        self.state = np.random.randint(0,self.arms)
        
    def get_reward(self,arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])
        
    def choose_arm(self, arm):
        reward = self.get_reward(arm)
        self.update_state()
        return reward

def model_and_loss_fn(input_dim, hidden_dim, output_dim):
    # Initialize model to use Sequential
    # Activation Function: Relu
    # input nodes=10, hidden nodes=100, output nodes=10
    model = Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.ReLU(),
    )
    # Mean Squared Error Loss Function
    loss_fn = nn.MSELoss()
    return model, loss_fn

# Creates column of zeros and updates the 
# current state based on the position
def set_new_pos(arms, pos, val=1):
    curent_pos_vec = np.zeros(arms)
    curent_pos_vec[pos] = val
    return curent_pos_vec

def softmax(av, tau=1.12):
    softm = ( np.exp(av / tau) / np.sum( np.exp(av / tau) ) )
    return softm

def train_model(env, arms, model, loss_fn, epochs=2500, learning_rate=1e-2):
    rewards = []
    # Get current state from environment
    cur_state = torch.Tensor(set_new_pos(arms,env.get_state())) 
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in range(epochs):
        
        # Run NN to generate reward predictions
        reward_pred = model(cur_state) 
        
        # Create probability distribution and normalize it
        av_softmax = softmax(reward_pred.data.numpy(), tau=2.0) 
        av_softmax /= av_softmax.sum() 
        
        choice = np.random.choice(arms, p=av_softmax) 
        cur_reward = env.choose_arm(choice) 
        one_hot_reward = reward_pred.data.numpy().copy() 
        one_hot_reward[choice] = cur_reward 
        reward = torch.Tensor(one_hot_reward)
        rewards.append(cur_reward)
        loss = loss_fn(reward_pred, reward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cur_state = torch.Tensor(set_new_pos(arms,env.get_state())) 
    return np.array(rewards)

def running_mean(x, num=50):
    length = x.shape[0] - num
    y = np.zeros(length)
    conv = np.ones(num)
    for i in range(length):
        y[i] = (x[i : i + num] @ conv) / num
    return y

def main():
    
    # Number of slot machines
    arms = 10
    # Nodes for NN
    input_dim, hidden_dim, output_dim = arms, 100, arms
    
    # Instantiate environment object 
    env = ContextBandit(arms)
    
    model, loss_fn = model_and_loss_fn(input_dim, hidden_dim, output_dim)
    
    # Train the model
    mean_rewards = train_model(env, arms, model, loss_fn)
    
    # Plot the results
    plt.xlabel("Plays",fontsize=10)
    plt.ylabel("Mean Rewards",fontsize=10)
    plt.plot(running_mean(mean_rewards,500))

if __name__ == "__main__":
    main()
