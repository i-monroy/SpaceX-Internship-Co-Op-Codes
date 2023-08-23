"""
Author: Isaac Monroy
Title: Frozen Lake Problem Solution
Description: 
    This program applies a Deep Q-Learning algorithm to solve
    the FrozenLake problem from the OpenAI Gym. The algorithm 
    uses a three-layered neural network to approximate the Q-value
    function. The network is trained using experiences stored in a 
    replay buffer to minimize correlations in observations. To 
    balance exploration and exploitation, we use an epsilon-greedy 
    policy. The code also includes functionality for saving and 
    loading the model.
"""
# Import necessary libraries
import gym  # OpenAI gym for the environment
import numpy as np  # Numpy for numerical operations
import tensorflow as tf  # TensorFlow for the deep learning framework
from tensorflow.keras.models import Sequential  # Keras Sequential model as a base for the Q-network
from tensorflow.keras.layers import Dense  # Keras Dense layer for creating the fully connected layers of the Q-network
from tensorflow.keras.optimizers import Adam  # Adam optimizer for training the Q-network
import random  # Random for epsilon-greedy action selection and minibatch selection
from collections import deque  # Deque data structure for the replay buffer

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Get state and action space sizes
state_space_size = env.observation_space.n
action_space_size = env.action_space.n

# Initialize Q-table to zeros
q_table = np.zeros((state_space_size, action_space_size))

# Define hyperparameters for Q-learning
num_episodes = 50000
max_steps_per_episode = 100

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

# List to hold all rewards from episodes for statistical purposes
rewards_all_episodes = []

# Define Deep Q-Network class
class DeepQNetwork:
    """Implements a multi-layer perceptron with Experience Replay"""
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        """Initialize variables, build model."""
        self.nS = states  # Number of states
        self.nA = actions  # Number of actions
        self.memory = deque(maxlen=2000)  # Experience replay memory to store experiences
        self.gamma = gamma  # Discount rate for future rewards
        self.epsilon = epsilon  # Exploration rate, to balance between exploration and exploitation
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate to decrease exploration over time
        self.model = self.build_model(alpha)  # Build the neural network model

    def build_model(self, alpha):
        """Build a 3-layer MLP with given learning rate."""
        model = Sequential()  # Sequential model is a linear stack of layers
        model.add(Dense(24, input_dim=self.nS, activation='relu'))
        model.add(Dense(24, activation='relu')) 
        model.add(Dense(self.nA, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=alpha))  # Use MSE loss and Adam optimizer for compilation
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))  # Append the experience tuple to the memory

    def state_to_one_hot(self, state):
        """One-hot encode state."""
        one_hot_state = np.zeros(self.nS)
        one_hot_state[state] = 1
        return np.array([one_hot_state])  # Return the one-hot encoded state

    def act(self, state):
        """Select action according to ε-greedy policy."""
        if np.random.rand() <= self.epsilon:  # If random number is less than or equal to epsilon, choose a random action (exploration)
            return random.randrange(self.nA)
        act_values = self.model.predict(state)  # If not, predict the Q-values of the state using the model
        return np.argmax(act_values[0])  # Select the action with the highest Q-value (exploitation)

    def replay(self, batch_size):
        """Update network weights using replay buffer."""
        minibatch = random.sample(self.memory, batch_size) 
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))  # Update the target for training
            target_f = self.model.predict(state)  # Predict the Q-value of the state
            target_f[0][action] = target  # Update the target Q-value for the action
            states.append(state[0])
            targets_f.append(target_f[0])
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:  # If epsilon is greater than its minimum value
            self.epsilon *= self.epsilon_decay  # Decay the epsilon to reduce exploration over time
        return loss

    def load(self, name):
        """Load weights from disk."""
        self.model.load_weights(name)  # Load the model weights from a file

    def save(self, name):
        """Save weights to disk."""
        self.model.save_weights(name)  # Save the model weights to a file

# DQN parameters
alpha = 0.001
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32

# Initialize DQN
dqn = DeepQNetwork(env.observation_space.n, env.action_space.n, alpha, gamma, epsilon, epsilon_min, epsilon_decay)

# DQN training algorithm
n_episodes = 10000
done = False
for e in range(n_episodes): 
    state = env.reset()  # Reset the environment to start a new episode
    state = dqn.state_to_one_hot(state)  # One-hot encode the state for the neural network input
    for time in range(5000):
        action = dqn.act(state)  # Select an action using the current policy (epsilon-greedy)
        next_state, reward, done, info = env.step(action)  # Execute the action in the environment
        reward = reward if not done else -10  # Penalize the agent by -10 if the episode is finished
        next_state = dqn.state_to_one_hot(next_state)  # One-hot encode the next state for the neural network input
        dqn.remember(state, action, reward, next_state, done)  # Store the experience in the replay buffer
        state = next_state  # Update the state to the next state
        if done: 
            print(f"episode: {e}/{n_episodes}, score: {time}, e: {dqn.epsilon:.2}") 
            break 
        if len(dqn.memory) > batch_size:
            loss = dqn.replay(batch_size)  # Sample a batch of experiences from the replay buffer and train the network
            if time % 10 == 0:
                print(f"episode: {e}/{n_episodes}, time: {time}, loss: {loss:.4f}")