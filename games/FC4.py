import numpy as np
import gym
from gym.spaces import Discrete
from gym.utils import seeding

class Agent():
    def __init__(self, id):
        super(Agent, self).__init__()
        self.id = id
        self.scripted = False

class FC4(gym.Env):
    def __init__(self, horizon, hunt_payoff=-2.0, gift_value=5.0, gifters='none'):
        self.horizon = horizon
        self.np_random, seed = seeding.np_random(0)
        self.agents = [Agent(i) for i in range(4)]
        self.n = 4
        self.hunt_payoff = hunt_payoff
        self.gift_value = gift_value
        self.gifters = gifters # "none" or "all"
        self.reset()
        self.payoffs = np.array([[[2,2],[self.hunt_payoff,1]],[[1,self.hunt_payoff],[1,1]]])
        
    def reset(self):
        self.t = 0
        #return [np.array([0,0,0,0]),np.array([0,0,0,0])]
        return [np.array([0.0]), np.array([0.0]), np.array([0.0]),np.array([0.0])] 
        
    def step(self, actions): # 1 is dare, 0 is chicken out

        self.t += 1
        # Each agent plays 4 games
        bool_R = np.identity(4,dtype=bool)
        R = np.zeros((4,4))
        for i in range(4):
            for j in range(4):
                # i and j have not played yet
                if not bool_R[i,j]:
                    a_i, a_j = actions[i], actions[j]
                    # reward i recieves playing j                
                    baseRew = self.payoffs[a_i%2,a_j%2][0]
                    temp = (a_i//2) - (a_j//2)
                    # you lose reward if you gifted
                    baseRew -= temp*self.gift_value
                    bool_R[i,j] = True
                    R[i,j] = baseRew
                    # reward j recieves playing i                
                    baseRew = self.payoffs[a_i%2,a_j%2][1]
                    temp = (a_j//2) - (a_i//2)
                    # you lose reward if you gifted
                    baseRew -= temp*self.gift_value
                    bool_R[j,i] = True
                    R[j,i] = baseRew
                    bool_R[j,i] = True
                    
                    
        rews = np.sum(R, axis=1)/3
        done = self.t >= self.horizon
        return [np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0])], rews, done, {}
    
    @property
    def action_space(self):
        if self.gifters == 'none':
            return [Discrete(2), Discrete(2), Discrete(2), Discrete(2)]
        elif self.gifters == 'all':
            return [Discrete(4), Discrete(4), Discrete(4), Discrete(4)]
        else:
            print("INVALID GIFTERS VALUE")
            exit()

    @property
    def observation_space(self):
        return [Discrete(2), Discrete(2), Discrete(2), Discrete(2)]
        #return [MultiDiscrete([2, 2, 2, 2]), MultiDiscrete([2, 2, 2, 2])]
        
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]