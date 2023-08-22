"""
Author: Isaac Monroy
Project Title: Deep Q-Networks Gridworld Algorithm
Description:
    This algorithm simulates a player learning to navigate a Gridworld 
    environment using Deep Q-Networks (DQNs). In the Gridworld, the 
    player (P) aims to reach the goal ("+") within a grid while avoiding 
    obstacles ("-"). The grid might look like this:

                [['+' '-' ' ' 'P']
                 [' ' 'W' ' ' ' ']
                 [' ' ' ' ' ' ' ']
                 [' ' ' ' ' ' ' ']]

    The player can move in four directions (up, down, left, and right). 
    The DQN is trained to generate Q-values for each action at each state. 
    This enables the player to take actions leading to the goal, eventually 
    winning the game in various scenarios. The model consists of 5 layers 
    (input, two hidden layers, and output) and utilizes a target network for 
    stable learning.
"""

# Import neccessary modules
from Gridworld import Gridworld # Handle the game's environment.
import numpy as np # For numerical operations
import torch # Used for defining and training the neural network model.
import torch.nn as nn # Used for building neural network layers and defining loss functions.
from torch import optim # Contains the Adam optimizer used to update the network's weights.
import random # Utilized for stochastic elements in the training process.
from IPython.display import clear_output # Used to clear output when printing loss during training.
import copy # To create a deep copy of the model for the target network.
from collections import deque # To handle the experience replay memory.

# Neural Network Model
class Net(nn.Module):
    def __init__(self, in_dim, feature_dim1, feature_dim2, out_dim):
        super (Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, feature_dim1),
            nn.ReLU(),
            nn.Linear(feature_dim1, feature_dim1),
            nn.ReLU(),
            nn.Linear(feature_dim1, feature_dim2),
            nn.ReLU(),
            nn.Linear(feature_dim2, out_dim),
            )

    def forward(self, inputs):
        return self.classifier(inputs)  

# Hyperparameters and Model Initialization
in_dim, feature_dim1, feature_dim2, out_dim = 64, 150, 100, 4
model = Net(in_dim, feature_dim1, feature_dim2, out_dim)
model2 = copy.deepcopy(model)
model2.load_state_dict(model.state_dict())
loss_fn = nn.MSELoss()
learning_rate = 1e-3
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
gamma, epsilon = 0.9, 0.3
action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
epochs, mem_size, batch_size, max_moves, sync_freq = 5000, 1000, 200, 50, 500
losses = []
replay = deque(maxlen=mem_size)
h = j = 0

for i in range(epochs):
    
    # Start a new game of random for each iteration
    game = Gridworld(size=4, mode='random')
    
    # Extract state information and add noise, then make pytorch var
    state1_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
    state1 = torch.from_numpy(state1_).float()
    mov = 0
    while(True):
        j += 1
        mov += 1
        # Q-network
        qval = model(state1)
        qval_ = qval.data.numpy()
        
        # Select action with epsilon greedy method
        # Return a random action to perform
        # Or choose the action with the highest Q value
        if (random.random() < epsilon):
            action_ = np.random.randint(0,4)
        else:
            action_ = np.argmax(qval_)
        action = action_set[action_]
        game.makeMove(action)
        
        # After making a move, get state of new game
        state2_ = game.board.render_np().reshape(1,64) + np.random.rand(1,64)/100.0
        state2 = torch.from_numpy(state2_).float()
        reward = game.reward()
        done = True if reward > 0 else False
        
        # Create experience and store it
        exp =  (state1, action_, reward, state2, done)
        replay.append(exp) 
        state1 = state2
        
        # Separate components of each experience into separate tensors
        if len(replay) > batch_size:
            minibatch = random.sample(replay, batch_size)
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch])
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
            
            # Recompute Q value from batch var
            Q1 = model(state1_batch) 
            # Recompute Q value but no gradients
            with torch.no_grad():
                Q2 = model2(state2_batch) 
            
            # Compute target Q values
            Y = reward_batch + gamma * ((1-done_batch) * torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X, Y.detach())
            print(i, loss.item())
            clear_output(wait=True)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            # Copy the model's parameters to model2, target network
            if j % sync_freq == 0: 
                model2.load_state_dict(model.state_dict())
        
        # Restart when game is over
        if reward != -1 or mov > max_moves:
            mov = 0
            break
        
losses = np.array(losses)

def test_model(model):
    # Start a new game of random for each iteration
    test_game = Gridworld(mode='random')
    
    # Extract state information and add noise, then make pytorch var
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    
    # status of game and move variables
    status = 1
    mov_t = 0
    while(status == 1): 
        
        # Q-network
        qval = model(state)
        qval_ = qval.data.numpy()
        
        # Choose action with the highest Q value
        action_ = np.argmax(qval_) 
        action = action_set[action_]
        test_game.makeMove(action)
        
        # After making a move, get state of new game
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        reward = test_game.reward()
        if reward != -1:
            status = 2 if reward > 0 else 0
        
        # Too many moves, game over
        mov_t += 1
        if (mov_t > 15):
            break
    
    win = True if status == 2 else False
    return win

# Play 1000 games and show how many games were won and win percentage
max_games = 1000
wins = 0
for i in range(max_games):
    win = test_model(model)
    if win:
        wins += 1
win_perc = float(wins) / float(max_games)
print("Games played: {0}, # of wins: {1}".format(max_games, wins))
print("Win percentage: {:.2f}%".format(100.0 * win_perc))