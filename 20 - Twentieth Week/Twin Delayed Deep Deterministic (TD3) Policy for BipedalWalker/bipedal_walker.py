"""
Author: Isaac Monroy
Title: Twin Delayed Deep Deterministic (TD3) Policy for BipedalWalker
Description:
    This script implements the Twin Delayed Deep Deterministic (TD3) 
    algorithm for training a BipedalWalker agent. TD3 is a reinforcement
    learning algorithm that is a variant of Deep Deterministic Policy 
    Gradients (DDPG) with additional tricks to deal with problems like 
    function approximation error. It uses two Q-functions to reduce 
    overestimation bias and delayed policy updates to reduce variance. 
    The agent uses an actor-critic architecture with the PyTorch library
    for defining and training the networks. The OpenAI gym's BipedalWalker-v3 
    environment is used as a training ground for the agent.
"""
# Import necessary libraries
import torch # Used for creating and training the neural networks used in the TD3 algorithm.
import torch.nn as nn # Used for creating the actor and critic networks.
import torch.nn.functional as F # Used for implementing activation functions.
import torch.optim as optim # Used for defining the optimizer that updates the parameters of the neural networks.
import numpy as np # Used for numerical operations like generating random numbers for exploration.
import gym # Used to create the BipedalWalker-v3 environment.

# Check for a GPU and set the device accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Actor network
class Actor(nn.Module):
    """
    This is the policy network, outputs the actions given states.
    """
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400) # First layer
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim) # Output layer

        self.max_action = max_action

    def forward(self, state):
        """
        Forward pass through the network given a state. Returns the 
        network's output action.
        """
        # Pass state as input through the following layers
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # Output layer & scale result by max action value
        a = torch.tanh(self.l3(a)) * self.max_action
        return a

# Define the Critic network
class Critic(nn.Module):
    """
    This is the value network, evaluates the actions given states and actions.
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400) # First layer
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1) # Output layer

    def forward(self, state, action):
        """
        Forward pass through the network given a state and action. Returns the 
        network's Q-value prediction.
        """
        # Concatenate state and action as input to the network
        state_action = torch.cat([state, action], 1)
        q = F.relu(self.l1(state_action)) # First layer
        q = F.relu(self.l2(q))
        q = self.l3(q) # Output layer
        return q

# Define the TD3 (Twin Delayed DDPG) class
class TD3:
    """
    Twin Delayed Deep Deterministic Policy Gradients, combines a policy network 
    (Actor) with two value networks (Critics). It also includes functionality to 
    perform training updates and action selection.
    """
    def __init__(self, lr, state_dim, action_dim, max_action):

        # Define the Actor model and target model
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Define the optimizer for the Actor model
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # Define the first Critic model and target model
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        # Define the optimizer for the first Critic model
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)

        # Define the second Critic model and target model
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        
        # Define the optimizer for the second Critic model
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.max_action = max_action

    def select_action(self, state):
        """
        Given a state, use the Actor network to select the next action.
        """
        # Reshape state to (1,-1) and convert to a tensor in the device
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        # Select an action by running the current state through the Actor network
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        """
        Update function to perform training updates. 
        """
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize first Critic
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize second Critic 
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates
            if i % policy_delay == 0:
                # Compute actor loss
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))

class ReplayBuffer:
    def __init__(self, max_size=int(5e5)):
        # Initialize an empty list to store the transitions
        self.buffer = []
        # Maximum size of the buffer
        self.max_size = max_size
        # Size of the buffer
        self.size = 0

    def add(self, transition):
        """
        Add a new transition to the buffer. A transition consists of a state, 
        action, reward, next state, and done signal.
        """
        # Increment the buffer size
        self.size += 1
        # Add the transition to the buffer
        # transition is a tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)

        # If buffer is full, remove 1/5th of the transitions
        if self.size > self.max_size:
            del self.buffer[0:int(self.max_size/5)]
            # Update the buffer size accordingly
            self.size = len(self.buffer)

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the buffer.
        """
        # Generate random indices for sampling from the buffer
        indexes = np.random.randint(0, self.size, size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []

        for i in indexes:
            # Get the corresponding transition for each index
            s, a, r, s_, d = self.buffer[i]

            # Append the elements of each transition to the corresponding list
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))

        # Return the elements as numpy arrays
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

# Hyperparameters 
env_name = "BipedalWalker-v3"
log_interval = 10           # Interval to print avg reward
random_seed = 0
gamma = 0.99                # Discount for future rewards
batch_size = 100            # Number of transitions from replay buffer
lr = 0.001
exploration_noise = 0.1 
polyak = 0.995              # Target policy update rate (1-tau)
policy_noise = 0.2          # Target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # Delayed policy updates parameter
max_episodes = 1000         # Max number of episodes
max_timesteps = 2000        # Max timesteps for one episode

# Initialize the gym environment
env = gym.make(env_name)

# Obtain state and action dimensions from the environment
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# Initialize policy and replay buffer
policy = TD3(lr, state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

# Set the random seed
env.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# logging variables
avg_reward = 0
ep_reward = 0

# Training loop
for episode in range(1, max_episodes+1):
    state = env.reset()

    for t in range(max_timesteps):
        # Select action and add exploration noise, then clip to the action space limits
        action = policy.select_action(state)
        action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)

        # Execute action in the environment and add to replay buffer
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state

        # Track rewards
        avg_reward += reward
        ep_reward += reward

        # Update policy if episode is done or at max timesteps
        if done or t==(max_timesteps-1):
            policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            break

    ep_reward = 0

    # Save and stop training if average reward threshold is reached
    if (avg_reward/log_interval) >= 300:
        print("########## Solved! ###########")
        break

    # Print average reward every log interval episodes
    if episode % log_interval == 0:
        avg_reward = int(avg_reward / log_interval)
        print("Episode: {}\tAverage Reward: {}".format(episode, avg_reward))
        avg_reward = 0

