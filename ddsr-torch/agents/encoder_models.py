import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
'''
this file contains models used as state encoder for representation learning 
'''
# ResNet code is taked from alpha-zero https://github.com/michaelnny/alpha_zero/blob/main/alpha_zero/core/network.py

def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w

def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)
    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ResNetBlock(nn.Module):
    """Basic redisual block."""
    def __init__(self, num_filters):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x):
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out
 

class ResNetEncoder(nn.Module):
    def __init__(self, obs_shape=(3, 84, 84), rep_dim=256, num_res_block=3, num_filters=64, num_padding=1, num_fc_units=256):
        super().__init__()

        c, h, w = obs_shape
        conv_out_hw = calc_conv2d_output((h, w), 3, 1, num_padding)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.conv_block = nn.Sequential(
                            nn.Conv2d(
                                in_channels=c,
                                out_channels=num_filters,
                                kernel_size=3,
                                stride=1,
                                padding=num_padding,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_features=num_filters),
                            nn.ReLU(),
                        )

        res_blocks_list = []
        for _ in range(num_res_block):
            res_blocks_list.append(ResNetBlock(num_filters))
            
        self.res_blocks = nn.Sequential(*res_blocks_list)
        
        self.linear_block = nn.Sequential(
                            nn.Conv2d(
                                in_channels=num_filters,
                                out_channels=2,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_features=2),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.Linear(2 * conv_out, rep_dim),
                        )

    def forward(self, obs):
        obs = obs / 255.
        print('obs.shape', obs.shape)
        conv_block_out = self.conv_block(obs)
        res_block_out = self.res_blocks(conv_block_out)
        rep = self.linear_block(res_block_out)
        print('conv_block_out.shape', conv_block_out.shape)
        print('res_block_out.shape', res_block_out.shape)
        print('rep.shape', rep.shape)
        return rep

    
class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape=(3, 84, 84), rep_dim=256):
        super().__init__()

        self.layers = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, rep_dim)),
            nn.ReLU(),
        )

    def forward(self, obs):
        rep = self.layers(obs)
        # print('rep.shape', rep.shape)
        return rep

class MLPEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim, rep_dim):
        super().__init__()
    
        self.layers = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim), 
                nn.ReLU(),
                nn.Linear(hidden_dim, rep_dim)
            )   
        
    def forward(self, x):
        rep = self.layers(x)
        return rep
    

# ---------- 

class VAEEncoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(VAEEncoder, self).__init__()
        
        encoding_dim = 32768
        self.enc1 = nn.Conv2d(in_channels=image_channels,out_channels=16, kernel_size=4, stride = 2, padding = 1)
        self.enc2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4,stride = 2, padding = 1)

        self.fc_mu    = nn.Linear(encoding_dim, latent_dim)
        self.fc_sigma = nn.Linear(encoding_dim, latent_dim)
        
    def forward(self, x):
        x_shapes = []
        x = F.gelu(self.enc1(x))
        x = F.gelu(self.enc2(x))
        x_shapes.append(x.shape)
        x   = x.view(-1, x_shapes[0][1]*x_shapes[0][2]*x_shapes[0][3])  
        
        mu  = self.fc_mu(x)
        logvar = torch.exp(self.fc_sigma(x))
        eps = torch.randn_like(logvar)
        z   = mu + (eps * logvar)
        
        return z, mu, logvar, x_shapes

class VAEDecoder(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(VAEDecoder, self).__init__()
 
        decoding_dim = 32768
        
        self.fc2  = nn.Linear(latent_dim, decoding_dim)
        self.dec1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=4, stride=2, padding=1)
        
    def forward(self, z, x_shapes):
        z = self.fc2(z)
        x = z.view(x_shapes[0])
        x = F.gelu(self.dec1(x))
        recon = torch.sigmoid(self.dec2(x))
        return recon

class ConvVAE(nn.Module):
    def __init__(self, image_channels, latent_dim):
        super(ConvVAE, self).__init__()
        self.encoder = VAEEncoder(image_channels, latent_dim)
        self.decoder = VAEDecoder(image_channels, latent_dim)

    def forward(self, x):
        z, mu, logvar, x_shapes = self.encoder(x)
        return self.decoder(z, x_shapes), mu, logvar
    
def vae_loss(VAE, x):
    x_hat, mu, logvar = VAE(x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    recon = ((x - x_hat)**2).sum() 
    VAE_loss = kld + recon
    return VAE_loss


# ----------

