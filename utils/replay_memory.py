import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    ''' Returns the overwritten transition in order to keep track of 
        replay buffer statistics 
    '''
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        overwritten_transition =  self.memory[self.position]
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        return overwritten_transition

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)