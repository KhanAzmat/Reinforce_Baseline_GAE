import os
import gym
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
writer = SummaryWriter("./tb_record_3")
        
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
        self.double()
        self.dropout = nn.Dropout(p=0.1)

        self.shared_first = nn.Linear(self.observation_dim, self.hidden_size) # initialize the shared layer
        self.shared_second = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_head = nn.Linear(self.hidden_size, self.action_dim) # initialize the action layer
        self.value_head = nn.Linear(self.hidden_size, 1) # initialize the value layer, since we only need one value, we set the output dimension to 1
        nn.init.kaiming_normal_(self.shared_first.weight) # random weight initialization
        nn.init.kaiming_normal_(self.shared_second.weight)
        nn.init.kaiming_normal_(self.action_head.weight)
        nn.init.kaiming_normal_(self.value_head.weight)
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        self.next_state = []
        self.done = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
        """

        input = F.relu(self.shared_first(state))
        input = self.dropout(input)
        input = F.relu(self.shared_second(input))
        input = self.dropout(input)
        action_prob = F.softmax(self.action_head(input), dim=-1) # softmax over the last dimension
        state_value = self.value_head(input) # no activation function for the value layer

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
        """
        state = torch.from_numpy(state).float() # convert the state to a tensor
        action_probability, state_value = self.forward(state) # get the action probability and the state value
        m = Categorical(action_probability) # create a categorical distribution over the list of probabilities of actions
        action = m.sample() # and sample an action using the distribution
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99, advantages=None):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
        """
        
        # Initialize the lists and variables

        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 

        vals = []

        for a, v in saved_actions:
            vals.append(v[0])

        for (log_prob, val), advantage in zip(saved_actions, advantages(self.rewards, vals, self.done)):
            policy_losses.append(-log_prob * advantage.detach()) # minimize the negative log likelihood
            value_losses.append(advantage.pow(2)) # maximize the advantage function
            # we need to maximize the advantage function, but we can minimize the negative of the advantage function
            # since the goal is to maximize the reward, we can minimize the negative of the reward
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def get_value(self, state):
        state = torch.from_numpy(state).float()
        _, state_value = self.forward(state)
        return state_value.item()

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):

    #Implement Generalized Advantage Estimation (GAE) for your value prediction

        # Initialize the lists and variables
        advantages = []
        advantage = 0
        next_value = 0 

        for reward, value, done in zip(reversed(rewards), reversed(values), reversed(done)):
            td_error = reward + self.gamma * next_value * (1 - done) - value # calculate the temporal difference error
            advantage = td_error + self.gamma * self.lambda_ * advantage * (1 - done) # calculate the advantage
            next_value = value # update the next value
            advantages.insert(0, advantage) # insert the advantage to the front of the list

        advantages = torch.tensor(advantages) # convert the list to a tensor
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5) # normalize the advantages

        return advantages


def train(lr=0.01, lam=0.99):
    """
        Train the model using SGD (via backpropagation)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    # using GAE as the value prediction
    value_prediction = GAE(gamma=0.99, lambda_=lam, num_steps=None)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    # scheduler = Scheduler.StepLR(optimizer, step_size=300, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        # scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        # GAE value prediction
        for t in range(10000):
            action = model.select_action(state)
            state_prime, reward, done, _ = env.step(action)
            model.rewards.append(reward)
            model.done.append(done)
            ep_reward += reward
            if done:
                break
            state = state_prime

        optimizer.zero_grad()
        loss = model.calculate_loss(0.99, value_prediction)
        writer.add_scalar('Loss', loss.item(), i_episode)
        loss.backward()
        optimizer.step()
        model.clear_memory()
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        writer.add_scalar('Reward', ep_reward, i_episode)
        writer.add_scalar('Length', t, i_episode)
        writer.add_scalar('Learning Rate', scheduler.get_lr()[0], i_episode)
        writer.add_scalar('EWMA Reward', ewma_reward, i_episode)

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/LunarLander-v2_{}_{}.pth'.format(lr, lam))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


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
    lr = 0.0005
    # lam = 0.5
    # lam = 0
    lam = 0.99 # lambda for GAE means as a trade-off between bias and variance, greater lambda means more bias and less variance
    env = gym.make('LunarLander-v2')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr, lam)
    test(f'LunarLander-v2_{lr}_{lam}.pth')
