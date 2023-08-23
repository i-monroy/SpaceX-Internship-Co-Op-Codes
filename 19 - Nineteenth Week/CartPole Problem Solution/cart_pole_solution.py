"""
Author: Isaac Monroy
Title: CartPole Problem Solution
Description: 
    This program applies a Deep Q-Learning algorithm to solve
    the CartPole balancing problem from the OpenAI Gym. The algorithm 
    uses a multi-layered neural network to approximate the Q-value
    function. The network is trained using experiences stored in a 
    replay buffer to minimize correlations in observations. To 
    balance exploration and exploitation, we use an epsilon-greedy 
    policy. Furthermore, the model's weights are updated periodically 
    to a target model to provide more stable learning targets. 
    The code also includes functionality for normalizing the rewards 
    using a StandardScaler from the sklearn library.
"""
# Import necessary libraries
import gym  # OpenAI gym for creating the CartPole environment
import numpy as np  # Numpy for numerical operations
import tensorflow as tf  # TensorFlow for creating and training the neural network
from tensorflow import keras  # Keras for creating the model structure
import random  # Random for epsilon-greedy action selection and sampling from the replay buffer
from sklearn.preprocessing import StandardScaler  # StandardScaler for reward normalization

# Create the environment
env = gym.make('CartPole-v1')
obs_space = env.observation_space.shape[0]
action_space = env.action_space.n

# Define the ReplayBuffer class which will be used to store experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # Set the capacity of the buffer
        self.buffer = []  # Initialize buffer
        self.position = 0  # Initialize position

    def push(self, state, action, reward, next_state, done):
        """
        Adds an experience to the buffer. If the buffer
        is full, the oldest experience is removed first.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity  # Cycle position

    def sample(self, batch_size):
        """Randomly samples experiences from the buffer."""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        """Returns the current size of internal memory."""
        return len(self.buffer)

# Define the neural network model
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[obs_space]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(action_space, activation='softmax')
])

# Set optimizer and loss function
optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.categorical_crossentropy

# Create replay buffer
replay_buffer = ReplayBuffer(10000)
batch_size = 32

# Create the target model and set its weights to match the model's
target_model = keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

scaler = StandardScaler()  # Initialize the scaler

# Run the episodes
for episode in range(n_episodes):
    state = env.reset()  # Reset the environment state
    episode_reward = 0
    for step in range(1, 1000):
        # Epsilon-greedy action selection
        if np.random.rand() <= epsilon:
            action = np.random.randint(action_space)  # Exploration: choose a random action
        else:
            # Exploitation: choose the best action (with highest probability) based on current policy
            state_for_model = tf.convert_to_tensor(state)
            state_for_model = tf.expand_dims(state_for_model, 0)
            probas = model(state_for_model)
            action = tf.argmax(probas[0]).numpy()

        # Perform the action and get the new state, reward and whether the episode is done
        next_state, reward, done, info = env.step(action)
        # Add the experience to the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # If there are enough experiences in the buffer, start training the model
        if len(replay_buffer) > batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            target_probas = model.predict_on_batch(states)

            # Use the target model to calculate future Q values for the next states
            next_probas = target_model.predict_on_batch(next_states)
            # Choose the maximum Q value for each sample for the Bellman update
            max_future_q = np.amax(next_probas, axis=1)

            # Normalize rewards using StandardScaler
            rewards = scaler.fit_transform(rewards.reshape(-1,1)).flatten()

            # Calculate the updated Q values for the Bellman update
            updated_qs = rewards + (1 - dones) * 0.95 * max_future_q

            # Update the target probabilities with the updated Q values
            target_probas[np.arange(batch_size), actions] = updated_qs

            # Train the model to minimize the loss between the current and target Q values
            with tf.GradientTape() as tape:
                current_probas = model(states)
                loss = tf.reduce_mean(loss_fn(target_probas, current_probas))

            # Calculate gradients
            grads = tape.gradient(loss, model.trainable_variables)
            # Apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Set the new state to the current state for the next iteration
        state = next_state
        # Increment the total reward for the episode
        episode_reward += reward
        if done:
            break

        # Copy the current policy to the target policy periodically
        if episode % 10 == 0:
            target_model.set_weights(model.get_weights())

    # Decrease epsilon for epsilon-greedy action selection
    epsilon *= epsilon_decay
    print('Episode: {}, Rewards: {}'.format(episode, episode_reward))