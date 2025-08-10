import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

'''
this file contains models used in auxiliary tasks such as dynamic model, reconstruction model, reward model, projection model, etc 
'''

class ProjectionDNN(nn.Module):
    '''
    A simple MLPs for projection F(x|k), where x and k are concatenated
    '''
    def __init__(self, rep_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(rep_dim*2, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, rep_dim)
            )    

    def forward(self, x, k):
        xk = torch.cat((x,k), 1)
        proj = self.layers(xk)
        return proj
    
class ProjectionLinear(nn.Module):
    '''
    A simple MLPs for projection F(x, k)=theta^T(x-k) where x and k are subtracted and theta is linear weights
    '''
    def __init__(self, rep_dim):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(rep_dim, rep_dim), 
            )    

    def forward(self, x, k):
        xk = x-k
        proj = self.layers(xk)
        return proj
    


    
class DeterministicTransitionModel(nn.Module):
    def __init__(self, rep_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc = nn. Linear(rep_dim + action_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, rep_dim)
  

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = None
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        return mu


class ProbabilisticTransitionModel(nn.Module):

    def __init__(self, rep_dim, action_dim, hidden_dim, max_sigma=1e1, min_sigma=1e-4):
        super().__init__()
        self.fc = nn. Linear(rep_dim + action_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, rep_dim)
        self.fc_sigma = nn.Linear(hidden_dim, rep_dim)

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        assert(self.max_sigma >= self.min_sigma)
  

    def forward(self, x):
        x = self.fc(x)
        x = self.ln(x)
        x = torch.relu(x)
        mu = self.fc_mu(x)
        sigma = torch.sigmoid(self.fc_sigma(x))  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma

    def sample_prediction(self, x):
        mu, sigma = self(x)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps