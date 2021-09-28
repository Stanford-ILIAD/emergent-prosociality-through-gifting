import random
import math
from itertools import count
import numpy as np
# torch
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
# helper functions
from utils.replay_memory import *
from utils.models import *



class DQN():
    '''
    This is the main class responsible for setting up the DQN with the 
    given paramters. 
    Inputs:
        - env 
            this is the multiagent env class that you pass through. 
        - alpha
            lr for the optimzier
        - hidden_sizes
            list of hidden units (does not include input output so 
            you can leave blank if you do not want any hidden layers)
        - batch_size
        - buffer_size
        - Gamma 
            Discount parameter used 
        - num_episodes 
            total number of episodes 
        - horizon
            time horizon for ea episode
        - epsilon_params
            a list containing [eps_start, eps_end, eps_decay]
        - target_update
            number of episodes we run before updating target
        - eval_rate
            number of episodes we run before evaluating (or saving)
    
    '''    
    def __init__(self,
                 env,
                 alpha,
                 hidden_sizes,
                 batch_size,
                 buffer_size,
                 Gamma,
                 num_episodes,
                 horizon,
                 epsilon_params,
                 target_update,
                 eval_rate):

        super(DQN, self).__init__()
        # First we init the inputs
        self.env = env
        self.alpha = alpha
        self.hidden_sizes = hidden_sizes
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.Gamma = Gamma
        self.num_episodes = num_episodes
        self.horizon = horizon
        self.epsilon_params = epsilon_params
        self.target_update = target_update
        self.eval_rate = eval_rate
        # Now define variables needed throughout simulation
        self.eval_out = []              # for storing outputs of self.eval_single_episode()
        self.steps_done = 0             # counter of steps iterated
        self.device = 'cpu'             
        self.policy_networks = []       # list of policy network classes for each agent
        self.target_networks = []       # list of target network classes for each agent
        self.optimizers = []            # list of optimizer classes for each agent
        self.replay_buffers = []        # list of replay buffers for each agent
        _obs = self.env.reset()         # using _obs to create appropriate size networks
        for i in range(env.n):
            policy_net = KLayerNN(input_size=len(_obs[i]),
                                hidden_sizes=self.hidden_sizes, 
                                output_size=self.env.action_space[i].n)
            target_net = KLayerNN(input_size=len(_obs[i]),
                                hidden_sizes=self.hidden_sizes, 
                                output_size=self.env.action_space[i].n)
            # policy_net.apply(init_weights) # Comment out to use default nn.Linear initialization
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval().to(self.device)
            policy_net.train().to(self.device)
            # optimizer = optim.SGD(policy_net.parameters(), lr=self.alpha)
            # optimizer = optim.RMSprop(policy_net.parameters(), lr=self.alpha)
            optimizer = optim.Adam(policy_net.parameters(), lr=self.alpha)
            self.policy_networks.append(policy_net)
            self.target_networks.append(target_net)
            self.optimizers.append(optimizer)
            # finally deal with buffer
            memory = ReplayMemory(self.buffer_size)
            self.replay_buffers.append(memory)

    def select_action_e_greedy(self, state_n):
        '''
        each agent follows e-greedy policy independently
        '''
        action_n = []
        # exponential epsilon decay computed below
        eps_threshold = (self.epsilon_params[1] + (self.epsilon_params[0] - self.epsilon_params[1])
        * math.exp(-1.0 * self.steps_done / self.epsilon_params[2]))
        self.steps_done += 1    # this function is called every step
        for agent_i in range(len(self.policy_networks)):
            sample = random.random()
            if sample < eps_threshold:
                # EXPLORE
                n_actions = self.env.action_space[agent_i].n
                action_n.append(torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long))
            else:
                # EXPLOIT
                with torch.no_grad():
                    state = torch.from_numpy(state_n[agent_i]).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                    action_n.append(self.policy_networks[agent_i](state).max(1)[1].view(1, 1))
        return action_n


    def optimize_model(self):
        '''
        preforms a single update step to the optimizers 
        '''
        for agent_i in range(self.env.n):
            # check if buffer is ready
            if len(self.replay_buffers[agent_i]) < self.batch_size:
                break 
            transitions = self.replay_buffers[agent_i].sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool) # Warning: if the entire sampled batch is a batch of next_state = None then this crashes
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. 
            state_action_values = self.policy_networks[agent_i](state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.batch_size, device=self.device)
            next_state_values[non_final_mask] = self.target_networks[agent_i](non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * self.Gamma) + reward_batch
            loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            # Optimize the model
            self.optimizers[agent_i].zero_grad()
            loss.backward()
            for param in self.policy_networks[agent_i].parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizers[agent_i].step()

    def eval_single_episode(self):
        '''
        Computes whatever you need for a single episode run, without updating anything. Returns avg_rewards
        '''
        state = self.env.reset()
        avg_rewards = np.zeros(self.env.n)
        done = False
        i = 0
        while not done:
            i += 1
            action_n = []
            for agent_i in range(self.env.n):
                state_i = torch.from_numpy(state[agent_i]).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                with torch.no_grad():
                    # Acts greedily during evaluation
                    action_i = self.policy_networks[agent_i](state_i).max(1)[1].view(1, 1)
                    action_n.append(action_i.item())
            state, rewards, done, _ = self.env.step(action_n)
            avg_rewards += np.asarray(rewards)
        avg_rewards = avg_rewards / self.horizon
        self.eval_out.append(avg_rewards)

    def train(self):
        '''
        main training function
        '''
        for i_episode in range(self.num_episodes):
            state_n = self.env.reset()
            # iterate through one episode
            for t in count():
                action_n = self.select_action_e_greedy(state_n)
                next_state_n, rewards, done, _ = self.env.step([action.item() for action in action_n])
                # Observe new state
                if not done:
                    pass
                else:
                    next_state_n = None
                for agent_i in range(self.env.n):   
                    reward_i = rewards[agent_i] # Reward for Agent i
                    reward_i = torch.tensor([reward_i], device=self.device).type(torch.FloatTensor)
                    # Store the transition in memory
                    state_i = torch.from_numpy(state_n[agent_i]).unsqueeze(0).type(torch.FloatTensor).to(self.device)
                    action_i = action_n[agent_i]
                    # really long line coming up, my baaaaad
                    next_state_i = torch.from_numpy(next_state_n[agent_i]).unsqueeze(0).type(torch.FloatTensor).to(self.device) if next_state_n is not None else None
                    self.replay_buffers[agent_i].push(state_i, action_i, next_state_i, reward_i)
                # Move to the next state
                state_n = next_state_n
                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break

            # evaluation and saving
            if i_episode % self.eval_rate == 0:
                self.eval_single_episode()

            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                i = 0
                for target_net in self.target_networks:
                    target_net.load_state_dict(self.policy_networks[i].state_dict())
                    i += 1

        