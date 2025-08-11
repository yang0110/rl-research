import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class StateValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512, emb_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value

class DiscreteQValueNet(nn.Module):
    def __init__(self, state_emb_dim, action_num, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(state_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_num)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s= torch.relu(self.fc2())
        q_value_list = self.fc3(s)
        return q_value_list

class SingleQValueNet(nn.Module):
    def __init__(self, state_emb_dim, action_emb_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(state_emb_dim + action_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        sa = torch.relu(self.fc1(sa))
        sa = torch.relu(self.fc2(sa))
        action_value = self.fc3(sa)
        return action_value

class ContinuousPolicy(nn.Module):
    def __init__(self, state_emb_dim, action_dim, hidden_dim=512):
        super(ContinuousPolicy, self).__init__()
        self.fc1 = nn.Linear(state_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Get mean and log_std from the network's output
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist

    def sample(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

class DiscretePolicy(nn.Module):
    def __init__(self, state_emb_dim, action_num, hidden_dim=512):
        super(DiscretePolicy, self).__init__()
        self.fc1 = nn.Linear(state_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, action_num)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.output_layer(x)
        dist = Categorical(logits=logits)
        return dist

    def sample(self, state):
        dist = self.forward(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

