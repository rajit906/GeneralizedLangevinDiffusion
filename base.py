from abc import ABC, abstractmethod
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionModel(ABC):
    """Abstract base class for diffusion models using PyTorch."""
    def __init__(self, name, gmm_params, T=1.0, n_steps=1000):
        self.name = name
        self.gmm_params = {k: torch.tensor(v, device=DEVICE, dtype=torch.float32) for k, v in gmm_params.items()}
        self.T = T
        self.n_steps = n_steps
        self.ts = torch.linspace(0, T, n_steps, device=DEVICE)
        self.dt = torch.tensor(T / n_steps, device=DEVICE, dtype=torch.float32)
        print(f"--- Initializing {self.name} ---")

    def _get_initial_samples(self, n_samples):
        """Samples from the GMM distribution."""
        counts = torch.multinomial(self.gmm_params['weights'], n_samples, replacement=True)
        samples = torch.cat([
            torch.randn(int((counts == i).sum()), device=DEVICE) * self.gmm_params['stds'][i] + self.gmm_params['means'][i]
            for i in range(len(self.gmm_params['weights']))
        ])
        return samples[torch.randperm(n_samples)]

    @abstractmethod
    def precompute(self):
        pass

    @abstractmethod
    def solve_forward_sde(self, x0):
        pass

    @abstractmethod
    def solve_reverse_sde(self, xT):
        pass

    @abstractmethod
    def run_demonstration(self, n_plot, n_hist):
        pass