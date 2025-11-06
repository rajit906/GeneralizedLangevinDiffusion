# models/mlp.py

import torch
import torch.nn as nn
from . import utils
import math

# --- Sinusoidal Time Embedding ---
# This class is needed by both MLP and ResNet
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,)
        device = t.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freq)
        emb = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

@utils.register_model(name='mlp')
class MLP(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        self.act = nn.SiLU()

        # --- Time Embedding ---
        self.time_emb_dim = 64 # You can make this configurable
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            self.act
        )
        
        # --- Network ---
        in_dim = input_dim + self.time_emb_dim # Input is (ps, t_emb)
        out_dim = input_dim # Output is score for (ps)

        self.main = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, t):
        # x is ps_inputs (B, 2)
        # t is time (B,)
        
        # 1. Embed time
        time_emb = self.time_embed(t) # (B, time_emb_dim)
        
        # 2. Concatenate x and embedded time
        h = torch.cat([x, time_emb], dim=1) # (B, 2 + time_emb_dim)
        
        # 3. Pass through network
        return self.main(h)

@utils.register_model(name='resnet')
class ResNet(nn.Module):
    def __init__(self,
                 config,
                 n_hidden_layers=4):
        super().__init__()
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        n_hidden_layers = config.n_hidden_layers
        self.act = nn.SiLU()
        self.n_hidden_layers = n_hidden_layers
        
        # --- Time Embedding ---
        self.time_emb_dim = 64 # Or add to config
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(self.time_emb_dim),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            self.act
        )
        
        in_dim = input_dim + self.time_emb_dim
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
        # x is ps_inputs (B, 2)
        # t is time (B,)
        
        # Embed time
        time_emb = self.time_embed(t) # (B, time_emb_dim)
        
        h = torch.cat([x, time_emb], dim=1) # (B, 2 + time_emb_dim)
        
        # Initial projection
        h = self.act(self.input_layer(h))

        # Residual blocks
        for layer in self.hidden_layers:
            residual = h
            h = self.act(layer(h))
            h = h + residual

        return self.output_layer(h)