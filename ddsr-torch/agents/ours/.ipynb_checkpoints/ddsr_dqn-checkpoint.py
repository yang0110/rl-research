import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from ddsr_models import ResNetEncoder, MLPEncoder, ImageEncoder, ProjectionNet, ValueNet, PolicyNetDis, PolicyNetCon

class DDSRAgent(object):
    """DDSR Algorithm."""
    def __init__(self, ):
    
    # modules
    self.encoder = 
    self.projector = 
    self.actor = 
    self.critic = 
    
    self.target_actor = 
    self.target_critic = 
    
    #optimizers
    
    

