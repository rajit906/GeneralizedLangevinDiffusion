import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from scipy.signal import convolve
from base import DiffusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneralizedLangevinDiffusion(DiffusionModel):
    """Implements the forward process for Generalized Langevin Diffusion."""
    def __init__(self, gmm_params, **kwargs):
        super().__init__('Generalized Langevin Diffusion', gmm_params, **kwargs)
        self.var_names = ['Position', 'Momentum', 'Memory (s)']
        self.beta = 4.0; self.gamma = 1.0; self.lmbda = 0.; self.c = 0.5
        self.M = 0.25 # Mass term, currently unused by the provided SDE
        # Set initial variances to match CLD: momentum variance = gamma_init * M
        self.s_inits = [0.01, 0.01] # p_init_var, s_init_var

    def precompute(self):
        print(f"Skipping pre-computation for {self.name} (not implemented)."); pass

    def solve_forward_sde(self, z0):
        print(f"Solving forward SDE for {self.name}...")
        zs=torch.zeros((z0.shape[0], self.n_steps, 3), device=DEVICE); zs[:,0,:]=z0
        sqrt_2beta = np.sqrt(2 * self.beta)
        M_inv = 1.0 / self.M
        for i in range(self.n_steps - 1):
            x, p, s = zs[:,i,0], zs[:,i,1], zs[:,i,2]
            dW = torch.randn(z0.shape[0], 2, device=DEVICE) * torch.sqrt(self.dt)
            dx =  (M_inv * self.beta) * p * self.dt
            dp = -self.beta * (x + self.gamma**2*M_inv*p + self.gamma*self.lmbda*self.c*s)*self.dt + sqrt_2beta*self.gamma*dW[:,0]
            ds = -self.beta * (self.gamma*self.lmbda*self.c*p + self.lmbda**2*s)*self.dt + sqrt_2beta*(self.lmbda*self.c*dW[:,0] + self.lmbda*np.sqrt(1-self.c**2)*dW[:,1])
            zs[:,i+1,0]=zs[:,i,0]+dx; zs[:,i+1,1]=zs[:,i,1]+dp; zs[:,i+1,2]=zs[:,i,2]+ds
        return zs

    def solve_reverse_sde(self, zT):
        print(f"Reverse SDE not implemented for {self.name}."); return None

    def run_demonstration(self, n_plot, n_hist):
        print(f"Running demonstration for {self.name}...")
        x0 = self._get_initial_samples(n_plot)
        z0_vars = [torch.randn(n_plot, device=DEVICE) * np.sqrt(s) for s in self.s_inits]
        z0 = torch.stack([x0] + z0_vars, dim=-1)

        forward_paths = self.solve_forward_sde(z0).cpu().numpy()
        # Create empty reverse paths for plotting
        reverse_paths = np.zeros_like(forward_paths)

        fig, axes = plt.subplots(3, 3, figsize=(18, 15)); fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        for i in range(3):
            axes[i, 0].plot(self.ts.cpu(), forward_paths[:, :, i].T, lw=1.5)
            axes[i, 0].set_title(f'Forward: {self.var_names[i]}'); axes[i, 0].set_ylabel(self.var_names[i])
            axes[i, 1].plot(self.ts.cpu(), reverse_paths[:, :, i].T, lw=1.5)
            axes[i, 1].set_title(f'Reverse: {self.var_names[i]} (NI)')
            if i == 2:
                axes[i, 0].set_xlabel('Time'); axes[i, 1].set_xlabel('Time'); axes[i, 2].set_xlabel('Value')

        plot_position_dist(reverse_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        plot_aux_dist(axes[1, 2], (reverse_paths[:, 0, 1], self.var_names[1]), target_dist=(0, np.sqrt(self.s_inits[0])))
        plot_aux_dist(axes[2, 2], (reverse_paths[:, 0, 2], self.var_names[2]), target_dist=(0, np.sqrt(self.s_inits[1])))

        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()