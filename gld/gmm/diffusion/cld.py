# TODO:
# Vectorize. Remove MVN. Remove inverse.
# Implement PFODE with inbuilt solver.
# Implement UBU

import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel
from scipy.integrate import solve_ivp
import scipy.linalg

DEVICE = torch.device("cpu")

class CriticallyDampedLangevin(DiffusionModel):
    """Implements Critically Damped Langevin Diffusion (CLD)."""
    def __init__(self, gmm_params, **kwargs):
        self.M = 1.
        self.Gamma = 2.0
        self.beta = 8.0 * np.sqrt(self.M)
        self.gamma_init = 1.
        v_init_var = self.gamma_init * self.M
        super().__init__('Critically Damped Langevin', gmm_params, **kwargs)
        self.v_init_var = v_init_var

        self.M_inv = 1.0 / self.M
        self.A = torch.tensor([
            [0, -self.M_inv],
            [1, self.Gamma * self.M_inv]
        ], dtype=torch.float32, device=DEVICE)

        self.G = torch.tensor([
            [0, 0],
            [0, np.sqrt(2 * self.Gamma * self.beta)]
        ], dtype=torch.float32, device=DEVICE)
        
        self.GGt = self.G @ self.G.T
        
        self.precompute()

    def precompute(self):
        # Pre-computation is simpler now. We only pre-compute the mean propagator.
        ts_np = self.ts.cpu().numpy()
        B_t = self.beta * ts_np
        
        M_ts_np = np.zeros((self.n_steps, 2, 2))
        exp_term1 = np.exp(-2 * B_t / self.Gamma)
        M_ts_np[:, 0, 0] = exp_term1 * (2 * B_t / self.Gamma + 1)
        M_ts_np[:, 0, 1] = exp_term1 * (4 * B_t / (self.Gamma**2))
        M_ts_np[:, 1, 0] = exp_term1 * (-B_t)
        M_ts_np[:, 1, 1] = exp_term1 * (-2 * B_t / self.Gamma + 1)

        self.M_ts = torch.from_numpy(M_ts_np).float().to(DEVICE)
        # We will compute the covariance dynamically below.

    def _get_perturbed_params(self, t_idx):
        # Get the pre-computed mean propagator
        M_t = self.M_ts[t_idx]
        
        # We will now compute the covariance dynamically for each GMM component
        weights = self.gmm_params['weights']
        means_t, covs_t = [], []

        for m, s in zip(self.gmm_params['means'], self.gmm_params['stds']):
            # --- Step 1: Compute the mean for this component ---
            mu0_k = torch.tensor([m, 0.], device=DEVICE)
            mean_t = M_t @ mu0_k
            means_t.append(mean_t)

            # --- Step 2: Compute the full covariance for this component ---
            # Initial conditions for this GMM component
            Sigma0_xx = s**2
            Sigma0_vv = self.gamma_init * self.M
            
            # Use the time t corresponding to t_idx
            t = self.ts[t_idx].item()
            B_t = self.beta * t
            
            # These are the full analytical formulas from Appendix B.1 of the paper
            exp_term_cov = np.exp(4 * B_t / self.Gamma)

            Sigma_t_xx = Sigma0_xx + exp_term_cov - 1 + (4*B_t/self.Gamma)*(Sigma0_xx - 1) + \
                         (4*B_t**2/self.Gamma**2)*(Sigma0_xx - 2) + (16*B_t**2/self.Gamma**4)*Sigma0_vv
            
            Sigma_t_xv = -B_t*Sigma0_xx + (4*B_t/self.Gamma**2)*Sigma0_vv - \
                         (2*B_t**2/self.Gamma)*(Sigma0_xx - 2) - (8*B_t**2/self.Gamma**3)*Sigma0_vv

            Sigma_t_vv = (self.Gamma**2/4)*(exp_term_cov - 1) + B_t*self.Gamma + \
                         Sigma0_vv*(1 + 4*B_t**2/self.Gamma**2 - 4*B_t/self.Gamma) + B_t**2*(Sigma0_xx - 2)

            # Assemble the covariance matrix and apply the outer exponential term
            cov_t_hat = torch.tensor([[Sigma_t_xx, Sigma_t_xv], [Sigma_t_xv, Sigma_t_vv]], device=DEVICE)
            cov_t = np.exp(-4 * B_t / self.Gamma) * cov_t_hat
            covs_t.append(cov_t)

        return weights, means_t, covs_t

    def _score_fn(self, z, t_idx):
        weights, means, covs = self._get_perturbed_params(t_idx)
        p_t_z = torch.zeros(z.shape[0], device=DEVICE)
        grad_v_p_t_z = torch.zeros(z.shape[0], device=DEVICE)
        for w, mean, cov in zip(weights, means, covs):
            stable_cov = cov + 1e-6 * torch.eye(2, device=DEVICE)
            dist = torch.distributions.MultivariateNormal(mean, stable_cov)
            pdf = torch.exp(dist.log_prob(z))
            grad_log_pdf = -torch.linalg.solve(stable_cov, (z - mean).T).T
            p_t_z += w * pdf
            grad_v_p_t_z += w * pdf * grad_log_pdf[:, 1]
            score_full = torch.zeros_like(z)
            score_full[:, 1] = grad_v_p_t_z / (p_t_z + 1e-8)
        return score_full

    def solve_forward_sde(self, z0):
        """Solves the forward SDE using matrix operations."""
        zs = torch.zeros(z0.shape[0], self.n_steps, 2, device=DEVICE)
        zs[:, 0, :] = z0
        sqrt_dt = torch.sqrt(self.dt)

        for i in range(self.n_steps - 1):
            z = zs[:, i, :]
            dW = torch.randn_like(z) * sqrt_dt
            drift = -self.beta * (self.A @ z.T).T
            diffusion = (self.G @ dW.T).T
            dz = drift * self.dt + diffusion
            zs[:, i+1, :] = z + dz
        return zs

    def solve_reverse_sde_em(self, zT):
        """
        Solves the reverse SDE using the Euler-Maruyama method.
        """
        zs = torch.zeros(zT.shape[0], self.n_steps, 2, device=DEVICE)
        zs[:, -1, :] = zT
        sqrt_dt = torch.sqrt(self.dt)
        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]
            score_full = self._score_fn(z, i)
            f_fwd = -self.beta * (self.A @ z.T).T
            score_drift = (self.GGt @ score_full.T).T
            drift_rev = -f_fwd + score_drift
            dW = torch.randn_like(z) * sqrt_dt
            diffusion = (self.G @ dW.T).T
            if i > 0:
                zs[:, i-1, :] = z - drift_rev * self.dt + diffusion
        return zs
    
    def solve_reverse_sde_sscs(self, zT):
        zs = torch.zeros(zT.shape[0], self.n_steps, 2, device=DEVICE); zs[:, -1, :] = zT
        M_inv = 1.0 / self.M
        B_half_dt = self.beta * (self.dt.item() / 2)
        exp_term = np.exp(-2 * B_half_dt / self.Gamma)
        exp_full_dt = np.exp(4 * B_half_dt / self.Gamma)
        cov_xx = exp_full_dt - 1 - 4*B_half_dt/self.Gamma - 8*B_half_dt**2/self.Gamma**2
        cov_xv = -4 * B_half_dt**2 / self.Gamma
        cov_vv = self.Gamma**2/4*(exp_full_dt-1) + B_half_dt*self.Gamma - 2*B_half_dt**2
        cov_half_np = np.array([[cov_xx, cov_xv], [cov_xv, cov_vv]]) * np.exp(-4*B_half_dt/self.Gamma)
        cov_half = torch.from_numpy(cov_half_np).float().to(DEVICE)

        for i in range(self.n_steps - 1, 0, -1):
            u_n = zs[:, i, :]
            mu_x_half = (2*B_half_dt/self.Gamma*u_n[:,0]-4*B_half_dt/self.Gamma**2*u_n[:,1]+u_n[:,0])*exp_term
            mu_v_half = (B_half_dt*u_n[:,0]-2*B_half_dt/self.Gamma*u_n[:,1]+u_n[:,1])*exp_term
            mu_half = torch.stack([mu_x_half, mu_v_half], dim=1)
            u_half = torch.distributions.MultivariateNormal(mu_half, cov_half).sample()
            score_v = self._score_fn(u_half, i)[:, 1]
            v_update = self.dt * (2 * self.beta * self.Gamma * (score_v + M_inv * u_half[:,1]))
            u_half_prime = u_half.clone(); u_half_prime[:, 1] += v_update
            mu_x_full = (2*B_half_dt/self.Gamma*u_half_prime[:,0]-4*B_half_dt/self.Gamma**2*u_half_prime[:,1]+u_half_prime[:,0])*exp_term
            mu_v_full = (B_half_dt*u_half_prime[:,0]-2*B_half_dt/self.Gamma*u_half_prime[:,1]+u_half_prime[:,1])*exp_term
            mu_full = torch.stack([mu_x_full, mu_v_full], dim=1)
            zs[:, i-1, :] = torch.distributions.MultivariateNormal(mu_full, cov_half).sample()
        return zs
    
    def solve_reverse_sde_ubu(self, zT):
        return None
    
    def solve_reverse_sde(self, zT, type='sscs'):
        if type=='em':
            return self.solve_reverse_sde_em(zT)
        elif type == 'sscs':
            return self.solve_reverse_sde_sscs(zT)
        elif type == 'ubu':
            return self.solve_reverse_sde_ubu(zT)

    def solve_pfode(self, zT, method="RK45"):
        """
        Solves the reverse process using the Probability Flow ODE with SciPy solvers.
        method: one of {"RK45", "RK23", "Radau", "BDF", "LSODA", "Euler"} 
        """
        z0 = zT.detach().cpu().numpy()

        def drift(t, z_flat):
            # Reshape back to (batch, 2)
            z = torch.tensor(z_flat.reshape(-1, 2), dtype=torch.float32, device=DEVICE)
            # compute score
            t_idx = int(np.clip(t / self.T * (self.n_steps - 1), 0, self.n_steps - 1))
            score_full = self._score_fn(z, t_idx)

            f_fwd = -self.beta * (self.A @ z.T).T
            score_drift = (self.GGt @ score_full.T).T
            drift_ode = -f_fwd + 0.5 * score_drift
            return drift_ode.cpu().numpy().flatten()

        # Integrate from T → 0
        sol = solve_ivp(
            drift, 
            t_span=[self.T, 0], 
            y0=z0.flatten(), 
            method=method, 
            t_eval=np.linspace(self.T, 0, self.n_steps)
        )

        zs = torch.tensor(sol.y.T.reshape(self.n_steps, -1, 2), dtype=torch.float32, device=DEVICE)
        return zs.permute(1, 0, 2)  # shape: (batch, n_steps, 2)
    
    def solve_forward_sde_sscs(self, n_samples):
        """
        Simulates forward SDE SSCS  by sampling from the perturbation kernels.
        """
        return None


    def run_demonstration(self, n_plot, n_hist):
        x0_plot = self._get_initial_samples(n_plot)
        z0_plot = torch.stack([x0_plot, torch.randn(n_plot, device=DEVICE) * np.sqrt(self.v_init_var)], dim=1)
        xT_hist = torch.randn(n_hist, device=DEVICE)
        vT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        zT_hist = torch.stack([xT_hist, vT_hist], dim=1)
        
        forward_paths = self.solve_forward_sde(z0_plot).cpu().numpy()
        reverse_sde_paths = self.solve_reverse_sde(zT_hist, type='sscs').cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        
        ts_cpu = self.ts.cpu()
        
        # --- Position Plots ---
        # Forward trajectories
        axes[0, 0].plot(ts_cpu, forward_paths[10:, :, 0].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[0, 0].plot(ts_cpu, forward_paths[:10, :, 0].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[0, 0].set_title('Forward: Position')
        axes[0, 0].set_ylabel('Position')
        axes[0, 0].set_ylim(-6, 6)

        # Reverse trajectories
        axes[0, 1].plot(ts_cpu, reverse_sde_paths[10:n_plot, :, 0].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[0, 1].plot(ts_cpu, reverse_sde_paths[:10, :, 0].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[0, 1].set_title('Reverse: Position')
        axes[0, 1].set_ylim(-6, 6)

        # Final distribution
        plot_position_dist(reverse_sde_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        axes[0, 2].set_title("Final Position Distribution")
        axes[0, 2].set_xlim(-6, 6)

        # --- Momentum Plots ---
        # Forward trajectories
        axes[1, 0].plot(ts_cpu, forward_paths[10:, :, 1].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[1, 0].plot(ts_cpu, forward_paths[:10, :, 1].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[1, 0].set_title('Forward: Momentum')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Momentum')
        axes[1, 0].set_ylim(-6, 6)

        # Reverse trajectories
        axes[1, 1].plot(ts_cpu, reverse_sde_paths[10:n_plot, :, 1].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[1, 1].plot(ts_cpu, reverse_sde_paths[:10, :, 1].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[1, 1].set_title('Reverse: Momentum')
        axes[1, 1].set_xlabel('Time')
        axes[1, 1].set_ylim(-6, 6)
        
        # Final distribution
        plot_aux_dist(axes[1, 2], (reverse_sde_paths[:, 0, 1], 'Momentum'), target_dist=(0, np.sqrt(self.v_init_var)))
        axes[1, 2].set_title("Final Momentum Distribution")
        axes[1, 2].set_xlim(-4, 4)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()