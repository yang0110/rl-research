''' 
state and state-action encoders
1. state = pixel encoder (atari, minigrid, dmc pixel)
3. state = vector encoder (dmc vector)
4. state-action = pixel + discrete action encoder (atari, minigrid)
4. state-action = pixel + continuous action encoder (dmc pixel)
5. state-action = vector + discrete action encoder (dmc vector)
'''

import torch
import torch.nn as nn

class PixelEncoder(nn.Module):
    '''
    channel = 4
    '''
    def __init__(self, env, emd_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, emb_dim),
        )

    def forward(self, x):
        return self.network(x / 255.0)

# ALGO LOGIC: initialize agent here:
class VectorEncoder(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512, emb_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        emb = self.fc3(x)
        return emb

class StateActionEncoder(nn.Module):
    '''
    first
    encode obs into embedding 
    encode action into embedding
    next
    concatenate both embeddings
    second
    pass through a MLP to get the final embedding
    '''
    def __init__(self, obs_dim, action_dim, hidden_dim=512, emb_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim+action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, o, a):
        oa = torch.cat([o, a], 1)
        oa = F.relu(self.fc1(oa))
        oa = F.relu(self.fc2(oa))
        emb = self.fc3(oa)
        return emb

# -- Residual Bloack for PixelEncoder --
def layer_init_normed(layer, norm_dim, scale=1.0):
    with torch.no_grad():
        layer.weight.data *= scale / layer.weight.norm(dim=norm_dim, p=2, keepdim=True)
        layer.bias *= 0
    return layer


def flatten01(arr):
    return arr.reshape((-1, *arr.shape[2:]))


def unflatten01(arr, targetshape):
    return arr.reshape((*targetshape, *arr.shape[1:]))


def flatten_unflatten_test():
    a = torch.rand(400, 30, 100, 100, 5)
    b = flatten01(a)
    c = unflatten01(b, a.shape[:2])
    assert torch.equal(a, c)


# taken from https://github.com/AIcrowd/neurips2020-procgen-starter-kit/blob/142d09586d2272a17f44481a115c4bd817cf6a94/models/impala_cnn_torch.py
class ResidualBlock(nn.Module):
    def __init__(self, channels, scale):
        super().__init__()
        # scale = (1/3**0.5 * 1/2**0.5)**0.5 # For default IMPALA CNN this is the final scale value in the PPG code
        scale = np.sqrt(scale)
        conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv0 = layer_init_normed(conv0, norm_dim=(1, 2, 3), scale=scale)
        conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = layer_init_normed(conv1, norm_dim=(1, 2, 3), scale=scale)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, scale):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3, padding=1)
        self.conv = layer_init_normed(conv, norm_dim=(1, 2, 3), scale=1.0)
        nblocks = 2  # Set to the number of residual blocks
        scale = scale / np.sqrt(nblocks)
        self.res_block0 = ResidualBlock(self._out_channels, scale=scale)
        self.res_block1 = ResidualBlock(self._out_channels, scale=scale)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

class PixelResidualEncoder(nn.Module):
    def __init__(self, env, out_dim=256, c=4, h=84, w=84):
        super().__init__()
        shape = (c, h, w)
        conv_seqs = []
        chans = [16, 32, 32]
        scale = 1 / np.sqrt(len(chans)) 
        for out_channels in chans:
            conv_seq = ConvSequence(shape, out_channels, scale=scale)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        encodertop = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=out_dim)
        encodertop = layer_init_normed(encodertop, norm_dim=1, scale=1.4)
        conv_seqs += [
            nn.Flatten(),
            nn.ReLU(),
            encodertop,
            nn.ReLU(),
        ]
        self.network = nn.Sequential(*conv_seqs)

    def forward(self, x):
        '''
        x: (batch_size, height, width, channels)
        '''
        x = x / 255.0
        x = x.permute(0, 3, 1, 2) # bhwc" -> "bchw"
        emb = self.network(x)
        return emb


