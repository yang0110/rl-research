import numpy as np
import torch.nn as nn

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_size = params/(1024*1024)
    print('model size = {:.2f}'.format(model_size), 'M')
    return model_size 


def indictor_sigmoid(x, beta=2):
    a = 1/(1+(np.exp((-beta*x))))
    y = 2*(1-a)
    return y 
           
    
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer