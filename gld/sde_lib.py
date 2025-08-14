# sde_lib.py

import torch
import numpy as np
from tqdm import tqdm

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, device='cuda'):
        """
        Construct a Variance Preserving SDE.
        
        Args:
            beta_min: a `float` for the minimum beta variance.
            beta_max: a `float` for the maximum beta variance.
            N: an `int` number of discretization steps.
            device: 'cuda' or 'cpu'
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.device = device
        self.t_span = torch.linspace(1e-5, 1.0, N, device=device)

        # Precompute alphas and betas
        self.betas = torch.linspace(beta_min, beta_max, N, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def marginal_prob(self, x_0, t):
        """
        Compute the mean and standard deviation of the perturbation kernel q(x_t | x_0).
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x_0
        std = torch.sqrt(1.0 - torch.exp(2. * log_mean_coeff))
        return mean, std[:, None]

    def sde(self, x, t):
        """
        Compute the drift and diffusion coefficients for the SDE.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    @torch.no_grad()
    def reverse_sde_sampler(self, model, shape, steps):
        """
        Sample from the reverse-time SDE using the Euler-Maruyama method.
        """
        x = torch.randn(shape, device=self.device)
        dt = 1. / steps
        
        for t_val in np.linspace(1, 1e-5, steps):
            t = torch.ones(shape[0], device=self.device) * t_val
            
            # Predict noise and convert to score
            predicted_noise = model(x, t)
            _, std = self.marginal_prob(torch.zeros_like(x), t)
            score = -predicted_noise / std
            
            # SDE update
            drift, diffusion = self.sde(x, t)
            drift = drift - diffusion[:, None]**2 * score
            z = torch.randn_like(x)
            x = x - drift * dt + diffusion[:, None] * torch.sqrt(torch.tensor(dt)) * z
            
        return x

    @torch.no_grad()
    def ode_sampler(self, model, shape, steps):
        """
        Sample from the Probability Flow ODE.
        """
        x = torch.randn(shape, device=self.device)
        dt = 1. / steps

        for t_val in np.linspace(1, 1e-5, steps):
            t = torch.ones(shape[0], device=self.device) * t_val
            
            # Predict noise and convert to score
            predicted_noise = model(x, t)
            _, std = self.marginal_prob(torch.zeros_like(x), t)
            score = -predicted_noise / std

            # ODE update
            drift, diffusion = self.sde(x, t)
            drift = drift - 0.5 * diffusion[:, None]**2 * score
            x = x - drift * dt

        return x