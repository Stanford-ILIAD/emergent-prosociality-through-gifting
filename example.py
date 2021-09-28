# This file will give an example of how to run 
# a DQN algorithm with gifting using our stag hunt environment
# Since all experiments will need many seeds, 
# we show the multiprocessing pipeline used

import os
import torch                 
from games.stag import stag          # environment class
from utils.dqn import DQN            # trainer class
import numpy as np
import random 
from multiprocessing import Pool     # for multiprocessing 

def run_seed(SEED, g, gifter):
    '''
    This function runs one seed of the stag hunt environment with DQN
    Inputs:
        SEED   - integer defining seed
        g      - payoff value for hunting alone ('r' or risk param in paper)
        gifter - string defining who has the ability to gift in the env
    Outputs: 
        partial_ratios - one-hot array shwoing if agents reached 
                        prosocial PNE/risk-dom PNE/not converged
    Note:
        Although the main train function -- DQN.train does not return anything, we store 
        the needed results in a list -- DQN.eval_out

        The function DQN.eval_single_episode(self) runs every DQN.eval_rate
        episodes. When called it runs a single episode
        of the environment without exploration, and appends the average
        rewards for each agent as a single array to DQN.eval_out
    '''
    # Seed
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # set default payoffs, as well as the hunt alone payoff
    h = 2
    m = 1
    p = 1
    payoffs = [h,m,p,g]
    # init the env relevant to this seed
    env = stag(horizon=10, payoffs=payoffs, gift_value=10, gifters=gifter)
    # init the archetecture we use to train our agents
    Arch = DQN(env=env,                                  # env class
                alpha=5e-4,                              # learning rate                   
                hidden_sizes=[],                         # no hidden layers used
                batch_size=64,                           
                buffer_size=100000,                     
                Gamma=0.99,                              # discount factor in Bellman Eq
                num_episodes=3001,                       # num of episodes to run for
                horizon=10,                              # of timesteps per episode
                epsilon_params=[0.3, 0.01, 20000],       # epsilon parameters for exp decay
                target_update = 250,                     # number of episodes between target updates
                eval_rate=3000)                          # this is set to 'num_episodes-1' so that 
                                                         # we only evaluate the final episode
    # initialize and train
    Arch.env.seed(SEED)
    Arch.train()          
    # get the last rewards evaluated                     
    rewards = Arch.eval_out[-1]
    # partial_ratios stores the result: 
    # [# reached prosocial PNE, # reached risk-dom PNE, # not converged]
    partial_ratios = [0,0,0]
    eps = 0.001
    if rewards[0]>(h-eps) and rewards[1]>(h-eps):
        # reached prosocial PNE
        partial_ratios[0] += 1
    elif rewards[0]>(m-eps) and rewards[0]<(m+eps) and rewards[1]>(m-eps) and rewards[1]<(m+eps):
        # reached risk-dom PNE
        partial_ratios[1] += 1
    else:
        # did not converge
        partial_ratios[2] += 1
    return partial_ratios 

# We will run 16 seeds with and without gifting for low and medium risk
def main():
    N = 16                         # number of seeds
    seeds = np.arange(N)
    gifters = ['neither','both']    # strings telling env if gifting is available
    risk    = [-2,-6]              # low and medium risk respectively

    # setup MP pipeline
    cpu_count = os.cpu_count()
    p = Pool(cpu_count)
    for g in risk:
        for gifter in gifters:
            print('Working on: risk, gifter: ',g,', ',gifter)
            args = []               # Pool needs a list of tuples for ea input
            for l in range(len(seeds)):
                args.append((int(seeds[l]),g,gifter))
            temp = p.starmap(run_seed,args)
            # sum all partial ratios
            ratios = np.sum(np.asarray(temp),0)
            if ratios[2] >0:
                print("WARNING: one or more SEEDs did not converge, consider running for longer")
            print('Results: ',ratios)


if __name__ == '__main__':
    main()