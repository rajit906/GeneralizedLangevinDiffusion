import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from scipy.signal import convolve
from base import DiffusionModel
from models import ScoreNetwork
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cpu")

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

    def solve_reverse_sde(self, xT, score_model=None):
        """
        Solve the reverse SDE, using either the ground truth score function
        or a trained score network if provided.
        """
        batch = xT.shape[0]
        xs = torch.zeros(batch, self.n_steps, device=DEVICE)
        xs[:, -1] = xT.view(-1)

        with torch.no_grad():  # <- disable gradient tracking
            for i in range(self.n_steps - 1, 0, -1):
                beta = self.beta_t[i]
                xt = xs[:, i]

                if score_model is None:
                    score = self._score_fn(xt, i)
                else:
                    t_idx = torch.full((batch,), i, device=DEVICE, dtype=torch.long)
                    score = score_model(xt, t_idx).view(-1)

                drift = -0.5 * beta * xt - beta * score
                noise = torch.randn_like(xt) * torch.sqrt(self.dt)
                xs[:, i-1] = xt - drift * self.dt + torch.sqrt(beta) * noise

        return xs



    def run_demonstration(self, n_plot, n_hist, score_model=None):
        x0_plot = self._get_initial_samples(n_plot)
        xT_hist = torch.randn(n_hist, device=DEVICE)
        
        forward_paths = self.solve_forward_sde(x0_plot).cpu().numpy()
        reverse_paths = self.solve_reverse_sde(xT_hist, score_model=score_model).cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        
        ts_cpu = self.ts.cpu()
        
        # Forward Process
        axes[0].plot(ts_cpu, forward_paths[10:].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[0].plot(ts_cpu, forward_paths[:10].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[0].set_title('Forward Process')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Position')
        
        # Reverse Process
        axes[1].plot(ts_cpu, reverse_paths[10:n_plot].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[1].plot(ts_cpu, reverse_paths[:10].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[1].set_title('Reverse Process')
        axes[1].set_xlabel('Time')
        
        # Final Distribution
        plot_position_dist(reverse_paths[:, 0], self.gmm_params, axes[2])
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def train_score_network_dsm(self, n_epochs=50, batch_size=128, lr=1e-3, n_steps=1000):
        """
        Train a score network with denoising score matching (DSM).
        Does not require access to ground truth score.
        """
        model = ScoreNetwork().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            for _ in range(n_steps):
                # sample x0 and timesteps
                x0 = self._get_initial_samples(batch_size).to(DEVICE).view(-1)   # (batch,)
                t_idx = torch.randint(0, self.n_steps, (batch_size,), device=DEVICE)

                # compute alpha and sigma for timestep
                alpha = self.alpha_t[t_idx].to(DEVICE)                 # (batch,)
                sqrt_alpha = torch.sqrt(alpha)
                sigma = torch.sqrt(1.0 - alpha)                        # (batch,)

                # corrupt data
                eps = torch.randn_like(x0)                             # (batch,)
                xt = sqrt_alpha * x0 + sigma * eps                     # (batch,)

                # target from DSM: -(eps / sigma)
                target = -(eps / sigma)

                # predict score
                scores_pred = model(xt, t_idx).view(-1)                # (batch,)

                # loss
                loss = loss_fn(scores_pred, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / n_steps
            losses.append(avg_loss)
            print(f"[DSM] Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")

        return model, losses


    def train_score_network_fisher(self, n_epochs=50, batch_size=128, lr=1e-3, n_steps=1000):
        """
        Vectorized training of score network (returns model, losses).
        This is Fisher matching where we have access to GT score.
        """
        model = ScoreNetwork().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0
            for _ in range(n_steps):
                # sample x0 and timesteps
                x0 = self._get_initial_samples(batch_size).to(DEVICE).view(-1)   # (batch,)
                t_idx = torch.randint(0, self.n_steps, (batch_size,), device=DEVICE)

                # compute alpha per-sample
                alpha = self.alpha_t[t_idx].to(DEVICE)                 # (batch,)
                sqrt_alpha = torch.sqrt(alpha)

                # sample x_t from p(x_t | x0) vectorized
                mean_t = x0 * sqrt_alpha                              # (batch,)
                std_t = torch.sqrt(1.0 - alpha)                       # (batch,)
                xt = mean_t + std_t * torch.randn_like(mean_t)        # (batch,)

                # compute ground-truth score s_t(x) = grad log p_t(x) vectorized
                numerator = torch.zeros_like(xt)
                denominator = torch.zeros_like(xt)
                # loop over mixture components (num components is small)
                for w, m, s in zip(self.gmm_params['weights'],
                                self.gmm_params['means'],
                                self.gmm_params['stds']):
                    # component parameters per sample
                    mean_comp = m * sqrt_alpha                        # (batch,)
                    std_comp = torch.sqrt((s**2) * alpha + (1.0 - alpha))  # (batch,)
                    dist = torch.distributions.Normal(mean_comp, std_comp)
                    pdf = torch.exp(dist.log_prob(xt))                # (batch,)
                    grad_log_pdf = -(xt - mean_comp) / (std_comp**2)  # (batch,)
                    numerator += w * pdf * grad_log_pdf
                    denominator += w * pdf

                scores_true = numerator / (denominator + 1e-8)        # (batch,)

                # predict, compute loss
                scores_pred = model(xt, t_idx)                        # (batch,)
                loss = loss_fn(scores_pred, scores_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / n_steps
            losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")

        return model, losses


