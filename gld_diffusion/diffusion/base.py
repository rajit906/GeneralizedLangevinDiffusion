from abc import ABC, abstractmethod
import torch

class DiffusionModel(ABC):
    """Abstract base class for diffusion models using PyTorch."""
    def __init__(self, name, data_dim, T=1.0, n_steps=1000):
        self.name = name
        self.data_dim = data_dim
        self.T = T
        self.n_steps = n_steps
        self.ts = torch.linspace(1e-4, T, n_steps) # Kept on CPU for numpy ops
        self.dt = torch.tensor(T / n_steps, dtype=torch.float32)
        print(f"--- Initializing {self.name} (Data Dim: {self.data_dim}) ---")

    @abstractmethod
    def precompute(self):
        """Precomputes acceleration structures (e.g., transition kernels)."""
        pass

    @abstractmethod
    def solve_reverse_sde_em(self, zT, score_model):
        """Solves the reverse-time SDE."""
        pass

    @abstractmethod
    def solve_pfode(self, zT, score_model):
        """Solves the reverse-time PFODE."""
        pass
    
    @abstractmethod
    def train_score_network(self, ScoreNetwork, dataloader, **kwargs):
        """Trains the score network."""
        pass