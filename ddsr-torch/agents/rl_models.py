import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

'''
This file contains models used in RL agents such as value (Q) function models, policy models 
'''

class ValueNet(nn.Module):
    '''
    return state value (V)
    '''
    def __init__(self, rep_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(rep_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),
            )    

    def forward(self, x):
        v_value = self.layers(x)
        return v_value
    

class PolicyNetDis(nn.Module):
    '''
    return Discrete action probability 
    '''
    def __init__(self, rep_dim, hidden_dim, action_num):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(rep_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_num),
            )    
    
    def forward(self, x):
        logits = self.layers(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, log_prob, entropy
    

class PolicyNetCont(nn.Module):
    '''
    return Continuous action probability 
    '''
    def __init__(self, rep_dim, hidden_dim, action_dim):
        super().__init__()
        self.mean_layer = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, actio_dim),
        )
        self.logstd_layer = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, x):
        action_mean = self.mean_layer(x)
        action_logstd = self.logstd_layer.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        return action, log_prob, entropy
    
class QValueNetCon(nn.Module):
    '''
    return state-action value (Q) of contionuous action
    '''
    def __init__(self, rep_dim, hidden_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(rep_dim+action_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Tanh(),
            )    
    
    def forward(self, x, a):
        assert x.size(0) == a.size(0)        
        xa = torch.cat([x, a], dim=1)
        q_value = self.layers(xa)
        return q_value    
    