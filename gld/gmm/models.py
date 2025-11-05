# models.py
import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,)
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -freq)
        emb = t.unsqueeze(1).float() * freqs.unsqueeze(0) * 1000.           # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)   # (batch, dim)
        return emb

class ScoreNetwork(nn.Module):
    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, hidden_dim),
            nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, t_idx):
        # Ensure x: (batch,) or (batch,1) -> (batch,1)
        x = x.view(-1, 1)
        # t_idx: (batch,)
        t_emb = self.time_emb(t_idx)            # (batch, time_emb_dim)
        t_h = self.time_mlp(t_emb)              # (batch, hidden_dim)
        h = torch.cat([x, t_h], dim=1)          # (batch, 1+hidden_dim)
        out = self.net(h)                       # (batch,1)
        return out.squeeze(-1)                  # (batch,)


class CLDScoreNetwork(nn.Module):
    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(1 + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, v, t_idx):
        # v: (batch, 1)
        # t_idx: (batch,)
        t_emb = self.time_emb(t_idx)   # (batch, time_emb_dim)
        inp = torch.cat([v, t_emb], dim=1)  # (batch, 1+time_emb_dim)
        return self.mlp(inp)  # (batch, 1)

class GLDScoreNetwork(nn.Module):
    """
    Score network for Generalized Langevin Diffusion.
    Inputs: (p, s) of shape (batch, 2), time index t_idx of shape (batch,)
    Outputs: score estimate of shape (batch, 2).
    """
    def __init__(self, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, ps, t):
        # ps: (B,2), t: (B,)
        t_emb = self.time_emb(t)
        h = torch.cat([ps, t_emb], dim=-1)
        return self.mlp(h)  # (B,2)