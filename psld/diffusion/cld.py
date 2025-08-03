import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CriticallyDampedLangevin(DiffusionModel):
    """Implements Critically Damped Langevin Diffusion (CLD)."""
    def __init__(self, gmm_params, **kwargs):
        self.M = 0.25
        self.Gamma = 1.0
        self.beta = 4.0
        self.gamma_init = 0.04
        v_init_var = self.gamma_init * self.M
        super().__init__('Critically Damped Langevin', gmm_params, **kwargs)
        self.v_init_var = v_init_var
        self.precompute()

    def precompute(self):
        print(f"Pre-computing {self.name} analytical moments...")
        ts_np = self.ts.cpu().numpy()
        B_t = self.beta * ts_np
        exp_term2 = np.exp(-4 * B_t / self.Gamma)

        M_ts_np = np.zeros((self.n_steps, 2, 2))
        exp_term1 = np.exp(-2 * B_t / self.Gamma)
        M_ts_np[:, 0, 0] = exp_term1 * (2 * B_t / self.Gamma + 1)
        M_ts_np[:, 0, 1] = exp_term1 * (4 * B_t / (self.Gamma**2))
        M_ts_np[:, 1, 0] = exp_term1 * (-B_t)
        M_ts_np[:, 1, 1] = exp_term1 * (-2 * B_t / self.Gamma + 1)

        Sigma0_vv = self.gamma_init * self.M
        Sigma_t_xx_added = (np.exp(4*B_t/self.Gamma)-1) + 4*B_t/self.Gamma*(-1) + 4*B_t**2/self.Gamma**2*(-2) + 16*B_t**2/self.Gamma**4*Sigma0_vv
        Sigma_t_xv_added = 4*B_t/self.Gamma**2*Sigma0_vv - 2*B_t**2/self.Gamma*(-2) - 8*B_t**2/self.Gamma**3*Sigma0_vv
        Sigma_t_vv_added = self.Gamma**2/4*(np.exp(4*B_t/self.Gamma)-1) + B_t*self.Gamma + Sigma0_vv*(1+4*B_t**2/self.Gamma**2-4*B_t/self.Gamma) + B_t**2*(-2)

        Sigma_ts_added_np = np.zeros((self.n_steps, 2, 2))
        Sigma_ts_added_np[:, 0, 0] = Sigma_t_xx_added * exp_term2
        Sigma_ts_added_np[:, 0, 1] = Sigma_ts_added_np[:, 1, 0] = Sigma_t_xv_added * exp_term2
        Sigma_ts_added_np[:, 1, 1] = Sigma_t_vv_added * exp_term2

        self.M_ts = torch.from_numpy(M_ts_np).float().to(DEVICE)
        self.Sigma_ts_added = torch.from_numpy(Sigma_ts_added_np).float().to(DEVICE)

    def _get_perturbed_params(self, t_idx):
        M_t = self.M_ts[t_idx]; Sigma_t_added = self.Sigma_ts_added[t_idx]
        means_t, covs_t = [], []

        for m, s in zip(self.gmm_params['means'], self.gmm_params['stds']):
            mu0_k = torch.tensor([m, 0.], device=DEVICE)
            Sigma0_k = torch.diag(torch.tensor([s**2, self.v_init_var], device=DEVICE))
            mean_t = M_t @ mu0_k
            cov_t = M_t @ Sigma0_k @ M_t.T + Sigma_t_added
            means_t.append(mean_t); covs_t.append(cov_t)
        return self.gmm_params['weights'], means_t, covs_t

    def _score_fn(self, z, t_idx):
        weights, means, covs = self._get_perturbed_params(t_idx)
        p_t_z = torch.zeros(z.shape[0], device=DEVICE)
        grad_v_p_t_z = torch.zeros(z.shape[0], device=DEVICE)
        for w, mean, cov in zip(weights, means, covs):
            dist = torch.distributions.MultivariateNormal(mean, cov)
            pdf = torch.exp(dist.log_prob(z))
            grad_log_pdf = -torch.linalg.solve(cov, (z - mean).T).T
            p_t_z += w * pdf
            grad_v_p_t_z += w * pdf * grad_log_pdf[:, 1]
        return grad_v_p_t_z / (p_t_z + 1e-8)

    def solve_forward_sde(self, z0):
        zs = torch.zeros(z0.shape[0], self.n_steps, 2, device=DEVICE); zs[:, 0, :] = z0
        M_inv = 1.0 / self.M
        for i in range(self.n_steps - 1):
            x, v = zs[:, i, 0], zs[:, i, 1]
            dx = (self.beta * M_inv) * v * self.dt
            dv = (-self.beta * x - self.beta * self.Gamma * M_inv * v) * self.dt + \
                 np.sqrt(2 * self.Gamma * self.beta) * torch.randn_like(v) * torch.sqrt(self.dt)
            zs[:, i+1, 0] = x + dx; zs[:, i+1, 1] = v + dv
        return zs

    def solve_reverse_sde(self, zT):
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
            score_v = self._score_fn(u_half, i)
            v_update = self.dt * (2 * self.beta * self.Gamma * (score_v + M_inv * u_half[:,1]))
            u_half_prime = u_half.clone(); u_half_prime[:, 1] += v_update
            mu_x_full = (2*B_half_dt/self.Gamma*u_half_prime[:,0]-4*B_half_dt/self.Gamma**2*u_half_prime[:,1]+u_half_prime[:,0])*exp_term
            mu_v_full = (B_half_dt*u_half_prime[:,0]-2*B_half_dt/self.Gamma*u_half_prime[:,1]+u_half_prime[:,1])*exp_term
            mu_full = torch.stack([mu_x_full, mu_v_full], dim=1)
            zs[:, i-1, :] = torch.distributions.MultivariateNormal(mu_full, cov_half).sample()
        return zs

    def run_demonstration(self, n_plot, n_hist):
        print(f"Running demonstration for {self.name}...")
        x0_plot = self._get_initial_samples(n_plot)
        z0_plot = torch.stack([x0_plot, torch.randn(n_plot, device=DEVICE) * np.sqrt(self.v_init_var)], dim=1)
        xT_hist = torch.randn(n_hist, device=DEVICE)
        vT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        zT_hist = torch.stack([xT_hist, vT_hist], dim=1)
        forward_paths = self.solve_forward_sde(z0_plot).cpu().numpy()
        reverse_paths = self.solve_reverse_sde(zT_hist).cpu().numpy()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10)); fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        axes[0, 0].plot(self.ts.cpu(), forward_paths[:, :, 0].T, lw=1.5); axes[0, 0].set_title('Forward: Position'); axes[0, 0].set_ylabel('Position')
        axes[0, 1].plot(self.ts.cpu(), reverse_paths[:n_plot, :, 0].T, lw=1.5); axes[0, 1].set_title('Reverse: Position')
        plot_position_dist(reverse_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        axes[1, 0].plot(self.ts.cpu(), forward_paths[:, :, 1].T, lw=1.5); axes[1, 0].set_title('Forward: Momentum'); axes[1, 0].set_xlabel('Time'); axes[1, 0].set_ylabel('Momentum')
        axes[1, 1].plot(self.ts.cpu(), reverse_paths[:n_plot, :, 1].T, lw=1.5); axes[1, 1].set_title('Reverse: Momentum'); axes[1, 1].set_xlabel('Time')
        plot_aux_dist(axes[1, 2], (reverse_paths[:, 0, 1], 'Momentum'))
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()