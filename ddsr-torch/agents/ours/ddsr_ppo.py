import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import ResNetEncoder, MLPEncoder, ImageEncoder, ProjectionNet, ValueNet, PolicyNetDis, PolicyNetCon

class DDSR_PPO(object):
    """DDSR + PPO Algorithm."""
    def __init__(self, ):
    
    # modules
    self.encoder = 
    self.projector = 
    self.actor = 
    self.critic = 
    
    self.target_actor = 
    self.target_critic = 
    
    #optimizers
    
    

