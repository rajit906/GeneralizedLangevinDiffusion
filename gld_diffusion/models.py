import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    """ Sinusoidal time embedding. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,)
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -freq)
        emb = t.unsqueeze(1).float() * freqs.unsqueeze(0) * 1000.
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1: # Handle odd dimensions
             emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class GLDScoreNetwork(nn.Module):
    """
    Generalized Score network for GLD.
    Inputs: 
      - (p, s) of shape (batch, 2 * data_dim)
      - time t of shape (batch,)
    Outputs: 
      - score estimate of shape (batch, 2 * data_dim).
    """
    def __init__(self, data_dim, hidden_dim=256, time_emb_dim=64):
        super().__init__()
        self.data_dim = data_dim
        self.input_dim = 2 * data_dim
        self.output_dim = 2 * data_dim
        
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, ps, t):
        # ps: (B, 2*d), t: (B,)
        t_emb = self.time_emb(t) # (B, time_emb_dim)
        h = torch.cat([ps, t_emb], dim=-1) # (B, 2*d + time_emb_dim)
        return self.mlp(h)  # (B, 2*d)
    
class Block(nn.Module):
    """Convolutional block with two convs, group norm, and SiLU."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.silu1 = nn.SiLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        self.silu2 = nn.SiLU()
        
        # Linear layer for time embedding
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

    def forward(self, x, t_emb):
        # x: (B, in_channels, H, W)
        # t_emb: (B, time_emb_dim)
        
        h = self.silu1(self.gn1(self.conv1(x)))
        
        # Add time embedding
        t_h = self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1) # (B, out_channels, 1, 1)
        h = h + t_h
        
        h = self.silu2(self.gn2(self.conv2(h)))
        
        return h
    
class Down(nn.Module):
    """Downsampling block."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.block = Block(in_channels, out_channels, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        return x, self.pool(x) # Return residual for skip connection

class Up(nn.Module):
    """Upsampling block."""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # in_channels will be 2 * out_channels (from skip + upsampled)
        self.block = Block(in_channels, out_channels, time_emb_dim)

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        return self.block(x, t_emb)
    
class GLDScoreUNet(nn.Module):
    """
    Score network for GLD on images (e.g., MNIST).
    Inputs: 
      - (p, s) concatenated on channel dim: (B, 2*C, H, W)
      - time t: (B,)
    Outputs: 
      - score estimate: (B, 2*C, H, W)
    """
    def __init__(self, in_channels=2, out_channels=2, time_emb_dim=64, base_dim=32):
        super().__init__()
        
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        
        # --- Encoder ---
        self.init_conv = nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1)
        
        self.down1 = Down(base_dim, base_dim, time_emb_dim)       # 28x28 -> 14x14
        self.down2 = Down(base_dim, base_dim * 2, time_emb_dim)   # 14x14 -> 7x7
        
        # --- Bottleneck ---
        self.mid = Block(base_dim * 2, base_dim * 4, time_emb_dim)
        
        # --- Decoder ---
        self.up1 = Up(base_dim * 6, base_dim * 2, time_emb_dim) # 7x7 -> 14x14
        self.up2 = Up(base_dim * 3, base_dim, time_emb_dim)     # 14x14 -> 28x28
        
        # --- Output ---
        self.final_conv = nn.Conv2d(base_dim, out_channels, kernel_size=1)

    def forward(self, ps, t):
        # ps: (B, 2C, H, W)
        # t: (B,)
        
        t_emb = self.time_emb(t)
        
        h = self.init_conv(ps)
        
        s1, h = self.down1(h, t_emb) # s1: 28x28, h: 14x14
        s2, h = self.down2(h, t_emb) # s2: 14x14, h: 7x7
        
        h = self.mid(h, t_emb)       # h: 7x7
        
        h = self.up1(h, s2, t_emb)   # h: 14x14
        h = self.up2(h, s1, t_emb)   # h: 28x28
        
        return self.final_conv(h)