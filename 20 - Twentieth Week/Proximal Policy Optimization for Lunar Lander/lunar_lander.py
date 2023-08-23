"""
Author: Isaac Monroy
Title: Proximal Policy Optimization for Lunar Lander
Description: 
    This script implements the Proximal Policy Optimization (PPO) 
    algorithm to train a Lunar Lander agent. PPO is a type of policy 
    gradient method for reinforcement learning, which uses a surrogate 
    objective to improve the policy. The agent uses two separate 
    networks, an actor and a critic, to estimate the policy and value 
    function, respectively. The script uses PyTorch to define and train
    the networks, and the OpenAI gym's LunarLander environment to train 
    the agent.
"""
import torch # Create and training the neural networks used in the PPO algorithm.
import torch.nn as nn # Used for creating the actor and critic networks.
import torch.optim as optim # Used for defining the optimizer that updates the parameters of the neural networks
import gym # Used to create the Lunar Lander environment.
from torch.distributions import Categorical # Used for creating a probability distribution over the possible actions

# Setting up device for GPU usage if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # Actor network defining policy parameterized by theta
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var), # input layer
                nn.Tanh(), # activation function
                nn.Linear(n_latent_var, n_latent_var), # hidden layer
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim), # output layer
                nn.Softmax(dim=-1) # output layer activation function
                )
        
        # Critic network for value function parameterized by phi
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var), # input layer
                nn.Tanh(), # activation function
                nn.Linear(n_latent_var, n_latent_var), # hidden layer
                nn.Tanh(),
                nn.Linear(n_latent_var, 1) # output layer
                )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        """
        Forward propagate input state through the actor
        network and sample an action from the output
        distribution.
        """
        state = torch.from_numpy(state).float().to(device) 
        action_probs = self.action_layer(state) # get action probabilities
        dist = Categorical(action_probs) # create a categorical distribution over the list of probabilities of actions
        action = dist.sample() # sample an action from the distribution
        
        # Store the state, action and log_prob of action
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()
    
    def evaluate(self, state, action):
        """
        Calculate the log probabilities of a batch of
        actions given states, and the entropy of the 
        action distribution and value of states.
        """
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        
        action_logprobs = dist.log_prob(action) # log prob of action given state
        dist_entropy = dist.entropy() # entropy of action distribution
        
        state_value = self.value_layer(state) # value of state
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        # Initializing actor and critic networks
        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=lr, betas=betas)
        # Making a copy of the policy to have a comparison 'old' network
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Defining the loss function
        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        """
        Update the policy network using Proximal Policy 
        Optimization.
        """
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # Calculate the total discounted reward
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Converting list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            # Final loss of PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

# Used for storing transitions
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

def main():
    #Hyperparameters 
    env_name = "LunarLander-v2"
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 20           # print avg reward in the interval
    max_episodes = 50000        # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    
    update_timestep = 2000      # update policy every n timesteps
    action_dim = 4
    state_dim = 8
    n_latent_var = 64           # number of variables in hidden layer
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    K_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    
    # Creating environment
    env = gym.make(env_name)
    if random_seed:
        env.seed(random_seed)
        torch.manual_seed(random_seed)
    
    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print(lr,betas)
    
    # Logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # Training loop
    for i_episode in range(1, max_episodes+1):
        state = env.reset()
        for t in range(max_timesteps):
            timestep += 1
            
            # Running policy_old
            action = ppo.policy_old.act(state, memory)
            state, reward, done, _ = env.step(action)
            
            # Save data in batch
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            
            # update if it's time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += reward
            if render:
                env.render()
            if done:
                break
        
        avg_length += t
        
        # Stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break
        
        # Log progress
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()