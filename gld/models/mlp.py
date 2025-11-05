# models/mlp.py

import torch
import torch.nn as nn
from . import utils

@utils.register_model(name='mlp')
class MLP(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        input_dim = config.input_dim
        index_dim = config.index_dim
        hidden_dim = config.hidden_dim

        act = nn.SiLU()
        in_dim = input_dim + index_dim
        out_dim = input_dim

        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, t):
        # Concatenate x and t
        h = torch.cat([x, t.reshape(-1, 1)], dim=1)
        return self.main(h)

@utils.register_model(name='resnet')
class ResNet(nn.Module):
    def __init__(self,
                 config,
                 n_hidden_layers=4):
        super().__init__()
        input_dim = config.input_dim
        index_dim = config.index_dim
        hidden_dim = config.hidden_dim
        n_hidden_layers = config.n_hidden_layers
        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers
        in_dim = input_dim + index_dim
        out_dim = input_dim

        # Input layer
        self.input_layer = nn.Linear(in_dim, hidden_dim)

        # Hidden layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden_layers)]
        )
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, t):
        # Embed time and concatenate
        time_emb = t.reshape(-1, 1)
        h = torch.cat([x, time_emb], dim=1)
        
        # Initial projection
        h = self.act(self.input_layer(h))

        # Residual blocks
        for layer in self.hidden_layers:
            residual = h
            h = self.act(layer(h))
            h = h + residual

        return self.output_layer(h)
