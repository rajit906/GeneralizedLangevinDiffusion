import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from scipy.signal import convolve
from base import DiffusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VPSDE(DiffusionModel):
    """Implements the first-order Variance Preserving SDE."""
    def __init__(self, gmm_params, beta_min=0.1, beta_max=20.0, **kwargs):
        super().__init__('VP-SDE', gmm_params, **kwargs)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.precompute()

    def precompute(self):
        self.beta_t = self.beta_min + self.ts * (self.beta_max - self.beta_min)
        self.alpha_t = torch.exp(-torch.cumsum(self.beta_t, dim=0) * self.dt)

    def _score_fn(self, x, t_idx):
        '''
        s_t(x) = ∇p_t(x) / p_t(x)
        p_t(x) = Σ w_k N(√a_t μ_{0,k}, a_t σ_{0,k}^2 + (1-a_t))
        '''
        alpha_t_val = self.alpha_t[t_idx:t_idx+1]
        p_t_x = torch.zeros_like(x)
        grad_p_t_x = torch.zeros_like(x)
        for w, m, s in zip(self.gmm_params['weights'], self.gmm_params['means'], self.gmm_params['stds']):
            mean_t = m * torch.sqrt(alpha_t_val)
            std_t = torch.sqrt(s**2 * alpha_t_val + (1 - alpha_t_val))
            dist = torch.distributions.Normal(mean_t, std_t)
            pdf = torch.exp(dist.log_prob(x))
            grad_log_pdf = -(x - mean_t) / std_t**2
            p_t_x += w * pdf
            grad_p_t_x += w * pdf * grad_log_pdf
        return grad_p_t_x / (p_t_x + 1e-8)

    def _perturbation_kernel(self, x, t_idx):
        """
        p_t(x_t|x_0) = N(x_0 * sqrt(alpha_t), (1 - alpha_t))
        """
        alpha_t_val = self.alpha_t[t_idx:t_idx+1]
        mean_t = x * torch.sqrt(alpha_t_val)
        std_t = torch.sqrt(1 - alpha_t_val)
        dist = torch.distributions.Normal(mean_t, std_t)
        x_t_sample = dist.rsample()
        return dist, x_t_sample

    def solve_forward_sde(self, x0):
        xs = torch.zeros(x0.shape[0], self.n_steps, device=DEVICE); xs[:, 0] = x0
        for i in range(self.n_steps - 1):
            beta = self.beta_t[i:i+1]
            drift = -0.5 * beta * xs[:, i]
            diffusion = torch.sqrt(beta)
            noise = torch.randn_like(xs[:, i]) * torch.sqrt(self.dt)
            xs[:, i+1] = xs[:, i] + drift * self.dt + diffusion * noise
        return xs

    def solve_reverse_sde(self, xT):
        xs = torch.zeros(xT.shape[0], self.n_steps, device=DEVICE); xs[:, -1] = xT
        for i in range(self.n_steps - 1, 0, -1):
            beta = self.beta_t[i:i+1]
            score = self._score_fn(xs[:, i], i)
            drift = -0.5 * beta * xs[:, i] - beta * score
            diffusion = torch.sqrt(beta)
            noise = torch.randn_like(xs[:, i]) * torch.sqrt(self.dt)
            xs[:, i-1] = xs[:, i] - drift * self.dt + diffusion * noise
        return xs

    def run_demonstration(self, n_plot, n_hist):
        print(f"Running demonstration for {self.name}...")
        x0_plot = self._get_initial_samples(n_plot); xT_hist = torch.randn(n_hist, device=DEVICE)
        forward_paths = self.solve_forward_sde(x0_plot).cpu().numpy()
        reverse_paths = self.solve_reverse_sde(xT_hist).cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5)); fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        axes[0].plot(self.ts.cpu(), forward_paths.T, lw=1.5); axes[0].set_title('Forward Process'); axes[0].set_xlabel('Time'); axes[0].set_ylabel('Position')
        axes[1].plot(self.ts.cpu(), reverse_paths[:n_plot].T, lw=1.5); axes[1].set_title('Reverse Process'); axes[1].set_xlabel('Time')
        plot_position_dist(reverse_paths[:, 0], self.gmm_params, axes[2])
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
