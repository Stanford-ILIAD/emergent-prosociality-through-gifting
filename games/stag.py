import numpy as np
import gym
from gym.spaces import Discrete
from gym.utils import seeding

class Agent():
    def __init__(self, id):
        super(Agent, self).__init__()
        self.id = id
        self.scripted = False

class stag(gym.Env):
    def __init__(self, horizon, payoffs, gift_value=5, gifters='neither'):
        self.horizon = horizon
        self.np_random, seed = seeding.np_random(0)
        self.agents = [Agent(i) for i in range(2)]
        self.n = 2
        self.h = payoffs[0]
        self.m = payoffs[1]
        self.p = payoffs[2]
        self.g = payoffs[3]
        self.gift_value = gift_value
        self.gifters = gifters # "neither", "both", "agent_0", or "agent_1"
        self.reset()
        
    def reset(self):
        self.t = 0
        #return [np.array([0,0,0,0]),np.array([0,0,0,0])]
        return [np.array([0.0]), np.array([0.0])] # always receive same state
        
    def step(self, actions): 

        self.t += 1
        a0, a1 = actions[0], actions[1]

        a0 = [a0%2, (a0//2)*self.gift_value]
        a1 = [a1%2, (a1//2)*self.gift_value]
        if a0[0] == 0 and a1[0] == 0:
            baseRew = [self.h,self.h]
        elif a0[0] == 0 and a1[0] == 1:
            baseRew = [self.g, self.p]
        elif a0[0] == 1 and a1[0] == 0:
            baseRew = [self.p, self.g]
        else:
            baseRew = [self.m,self.m]

        addRew = [a1[1] - a0[1], a0[1] - a1[1]]

        rews = np.array(baseRew) + addRew
        #obs = np.array([a0[0], a0[1], a1[0], a1[1]]) # observation for multidiscrete case
        done = self.t >= self.horizon
        return [np.array([0.0]), np.array([0.0])], rews, done, {}
    
    @property
    def action_space(self):
        if self.gifters == 'neither':
            return [Discrete(2), Discrete(2)]
        elif self.gifters == 'both':
            return [Discrete(4), Discrete(4)]
        elif self.gifters == 'agent_0':
            return [Discrete(4), Discrete(2)]
        elif self.gifters == 'agent_1':
            return [Discrete(2), Discrete(4)]
        else:
            print("INVALID GIFTERS VALUE")
            exit()

    @property
    def observation_space(self):
        return [Discrete(2), Discrete(2)]
        #return [MultiDiscrete([2, 2, 2, 2]), MultiDiscrete([2, 2, 2, 2])]
        
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]