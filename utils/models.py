import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KLayerNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        hidden_sizes = [input_size] + hidden_sizes + [output_size]
        self.layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = F.relu(x)

        x = self.layers[len(self.layers) - 1](x)
        return x

# def init_weights(layer):
#     '''
#     Initializes the weights in our network
#     '''
#     print('Initializing weights')
#     if type(layer) == nn.Linear:
#         nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
#         if layer.bias is not None: 
#             layer.bias.data.zero_()

def zero_weights(layer):
    '''
    Initializes the weights in our network
    '''
    # print('Initializing weights')
    if type(layer) == nn.Linear:
        with torch.no_grad():
            for i in range(layer.weight.shape[0]):
                layer.weight[i] = torch.zeros(1)
            for i in range(layer.bias.shape[0]):
                layer.bias[i] = torch.zeros(1)

def small_weights(layer):
    '''
    Initializes the weights in our network
    '''
    # print('Initializing weights')
    if type(layer) == nn.Linear:
        with torch.no_grad():
            for i in range(layer.weight.shape[0]):
                layer.weight[i] = torch.tensor(np.random.uniform(-0.1,0.1))
            for i in range(layer.bias.shape[0]):
                layer.bias[i] = torch.tensor(np.random.uniform(-0.1,0.1))
def init_weights(layer):
    '''
    Initializes the weights in our network
    '''
    print('Initializing weights')
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None: 
            layer.bias.data.zero_()

# def init_weights(layer):
#     '''
#     Initializes the weights in our network
#     '''
#     print('Initializing weights')
#     if type(layer) == nn.Linear:
#         torch.nn.init.zeros_(layer.weight)
#         if layer.bias is not None: 
#             layer.bias.data.zero_()