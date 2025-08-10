import torch
import torch.nn as nn

class StateValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512, emb_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        emb = self.fc3(x)
        return emb

class ActionValueNet(nn.Module):
    '''
    state_emb_dim = obs_dim 
    action_emb_dim = action_dim
    '''
    def __init__(self, obs_dim, action_dim, hidden_dim=512, emb_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, o, a):
        oa = torch.cat([o, a], dim=1)
        oa = torch.relu(self.fc1(oa))
        oa = torch.relu(self.fc2(oa))
        emb = self.fc3(oa)
        return emb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ContinuousPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ContinuousPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Output layers for mean and standard deviation
        self.mean_layer = nn.Linear(256, action_dim)
        # We learn the log of the standard deviation to ensure it's always positive.
        self.log_std_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Get mean and log_std from the network's output
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent it from becoming too large or small
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        # Create a Normal distribution
        dist = Normal(mean, std)
        return dist

# Example usage
state_dim = 4
action_dim = 2
policy = ContinuousPolicy(state_dim, action_dim)
state = torch.randn(1, state_dim)
dist = policy(state)

# Sample an action from the distribution
action = dist.sample()
# Calculate the log probability of the sampled action
log_prob = dist.log_prob(action).sum(dim=-1)

print(f"Sampled Action: {action}")
print(f"Log Probability: {log_prob}")


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Output layer for the logits of each action
        self.output_layer = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Get the logits for each action
        logits = self.output_layer(x)
        
        # Create a Categorical distribution using the logits
        dist = Categorical(logits=logits)
        return dist

# Example usage
state_dim = 4
action_dim = 3  # e.g., move left, move right, stay
policy = DiscretePolicy(state_dim, action_dim)
state = torch.randn(1, state_dim)
dist = policy(state)

# Sample an action from the distribution
action = dist.sample()
# Calculate the log probability of the sampled action
log_prob = dist.log_prob(action)

print(f"Sampled Action: {action}")
print(f"Log Probability: {log_prob}")