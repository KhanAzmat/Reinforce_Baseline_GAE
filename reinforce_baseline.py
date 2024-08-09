import os
import  gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_2")

class Value_Network(nn.Module):
    def __init__(self):
        super(Value_Network, self).__init__()

        self.observation_dim = env.observation_space.shape[0]
        self.hidden_size = 256
        

        self.linear1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, 1)

        # self.double()

    def forward(self,state):
        state = F.relu(self.linear1(state))
        state = self.linear2(state)

        return state

class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 256

        self.value_function = Value_Network()
        self.optimizer = optim.Adam(self.value_function.parameters(), lr=0.001)
        
        '''
        Simple neural network with softmax action selection
        '''
        
        self.layer1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.action_dim)

        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.states = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
        """
        state = self.layer1(state)
        state = F.relu(state)
        state = self.layer2(state)

        action_prob = F.softmax(state, dim=1)
        state_value = state

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
        """

        state_value = torch.from_numpy(state).float().unsqueeze(0)
        probs,_ = self.forward(state_value)
        m = Categorical(probs)
        action = m.sample()
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self,states, gamma=0.999):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []
        
        for r in self.rewards[::-1]:
            R = r + gamma * R
            returns.insert(0,R)
        
        returns = torch.tensor(returns)

        value_estimates = []

        for state in states:
            state = torch.from_numpy(state).float().unsqueeze(0)
            value_estimates.append(self.value_function(state))
        
        value_estimates = torch.stack(value_estimates).squeeze() # rewards to go

        value_loss = F.mse_loss(value_estimates.float(), returns.float())

        # update the weights
        self.optimizer.zero_grad()
        value_loss.backward()
        self.optimizer.step()

        #calculate advantage
        advantage = []
        for value, R in zip(value_estimates, returns):
            advantage.append(R-value)
        
        advantage = torch.Tensor(advantage)

        for (log_prob,value),adv in zip(self.saved_actions, advantage):
            policy_losses.append( -log_prob * adv)

        policy_loss = torch.stack(policy_losses).sum()

        loss = policy_loss
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)

    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        states = []
        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        for t in range(1, 10000):
            #Select action
            action = model.select_action(state)

            #Take action
            next_state, reward, done, _ = env.step(action)

            states.append(state)

            #Save Rerward 
            model.rewards.append(reward)

            #Move to the next state
            state = next_state

            #Update episode reward
            ep_reward += reward

            if done:
                break

        #Calculate loss
        loss = model.calculate_loss(states)

        #Perform Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Clear memory
        model.clear_memory()

        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Length', t, i_episode)
        writer.add_scalar('Ewma_Reward', ewma_reward, i_episode)
        writer.add_scalar('Loss', loss, i_episode)

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break
    
    writer.close()  # Close the Tensorboard writer


def test(name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.0003
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'LunarLander_{lr}.pth')
