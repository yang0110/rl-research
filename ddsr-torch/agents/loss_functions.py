import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from utils import indictor_sigmoid

'''
This file contains loss functions used by RL agent such as value loss, policy loss, entropy loss, kl loss, 
and 
auxiliary loss functions such as bismulation loss, reconstruction loss, dynamic loss, reward loss and successor representation loss
etc
'''

def get_rep_sum_loss(rep_x, rep_ks, projector):
    '''
    rep_x: 1 x rep_dim # representationof state x
    rep_ks: K x rep_dim # representation of a set of K states
    '''
    K = rep_ks.shape[0]
    repeat_rep_x = rep_x.repeat(K, 1)
    
    # DNN projector
    concat_rep = torch.cat((repeat_rep_x, rep_k), dim=1)
    projection = projector(concat_rep)
    
    # linear, rep_diff projector
    diff_rep = repeat_rep_x - rep_k 
    projection = projector(diff_rep)
    
    sum_projection = projection.sum(dim=0).detach()
    loss_fn = nn.MSELoss()
    
    loss = loss_fn(rep_x, sum_projection)
    return loss 
    

def get_sr_loss(rep_x, rep_k, rep_next_x, projector, gamma):
    '''
    successor representation learning loss
    '''
    loss_fn = nn.MSELoss()
    rep_norm = torch.linalg.vector_norm(rep_x-rep_k)
    target = indictor_sigmoid(rep_norm)+gamma*torch.linalg.vector(projector(rep_next_x, rep_k))
    pred = torch.linalg.vector(projector(rep_x, rep_k)
    loss = loss_fn(pred, target.detach())

    return loss
    