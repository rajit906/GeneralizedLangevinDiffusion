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
        self.beta = 1.0; self.gamma = 1.68; self.lmbda = 1.; self.c = 0.5
        self.M = 0.25
        self.s_inits = [0.01, 0.01]
        self.perturbation_cache = {}  # cache of (mean, cov) per time step

    def precompute(self):
        print("Precomputing full (x, p, s) perturbation covariances...")

        gamma, lmbda, c = self.gamma, self.lmbda, self.c
        beta = self.beta

        ts = self.ts.to('cpu')  # Time vector on CPU for stability
        N = len(ts)

        A = -beta * torch.tensor([
            [0, -1, 0],
            [1, gamma**2, gamma * lmbda * c],
            [0, gamma * lmbda * c, lmbda**2]
        ], dtype=torch.float32)  # [3, 3]

        B = np.sqrt(2 * beta) * torch.tensor([
            [0, 0],
            [gamma, 0],
            [lmbda * c, lmbda * np.sqrt(1 - c**2)]
        ], dtype=torch.float32)  # [3, 2]

        BBt = B @ B.T  # [3, 3]

        max_order = 5
        powers_A = [torch.eye(3)]
        for k in range(1, max_order + 1):
            powers_A.append(powers_A[-1] @ A)

        covs = torch.zeros((N, 3, 3))
        for t_idx in range(N):
            t = ts[t_idx]
            mu_t = expm(A * t) @ z0_mean # Use expm for cov as well if necessary.
            cov_t = torch.zeros(3, 3)
            for k in range(max_order + 1):
                Ak = powers_A[k]
                AkBBtAkT = Ak @ BBt @ Ak.T
                coeff = (t ** (2 * k + 1)) / np.math.factorial(2 * k + 1)
                cov_t += coeff * AkBBtAkT
            if t == 0.0:
                cov_t += 1e-4 * torch.eye(3)
            covs[t_idx] = cov_t

        for t_idx in range(N):
            self.perturbation_cache[t_idx] = {
                'mean': torch.zeros(3, device=DEVICE),
                'cov': covs[t_idx].to(DEVICE).detach().clone()
            }



    def _get_perturbed_params(self, t_idx):
        """Retrieves cached GMM mean and cov for conditional p_t(p, s | x)."""
        cached = self.perturbation_cache[t_idx]
        weights = [1.0]
        means_out = [cached['mean']]
        covs_out = [cached['cov']]
        return weights, means_out, covs_out


    def _score_fn(self, z, t_idx):
        """Computes the score \nabla_{p,s} log p_t(p,s | x) for 1D-GMM using exact Gaussian components."""
        weights, means, covs = self._get_perturbed_params(t_idx)
        x, p, s = z[:, 0], z[:, 1], z[:, 2]
        cond_scores = torch.zeros(z.shape[0], 2, device=DEVICE)  # [batch_size, 2] for (dp, ds)
        total_pdf = torch.zeros(z.shape[0], device=DEVICE)

        for w, mean, cov in zip(weights, means, covs):
            mu_x, mu_p, mu_s = mean[0], mean[1], mean[2]
            cov_ps = cov[1:, 1:]  # 2x2 submatrix for (p,s)
            mean_ps = torch.stack([p - mu_p, s - mu_s], dim=-1)  # [batch_size, 2]

            dist = torch.distributions.MultivariateNormal(torch.zeros(2, device=DEVICE), cov_ps)
            exponent = dist.log_prob(mean_ps)
            grad_log_pdf = -torch.linalg.solve(cov_ps, mean_ps.T).T

            weight_pdf = w * torch.exp(exponent)
            total_pdf += weight_pdf
            cond_scores += weight_pdf.unsqueeze(-1) * grad_log_pdf

        return cond_scores / (total_pdf.unsqueeze(-1) + 1e-8)  # normalize

    def solve_reverse_sde(self, zT):
        print(f"Solving reverse SDE for {self.name}...")
        if not self.perturbation_cache:
            print(f"Precomputing perturbations for {self.name}...")
            self.precompute()
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, 0, :] = zT

        gamma, lmbda, c, beta = self.gamma, self.lmbda, self.c, self.beta

        Gamma = torch.tensor([
            [gamma**2, gamma * lmbda * c],
            [gamma * lmbda * c, lmbda**2]
        ], dtype=torch.float32, device=DEVICE)  # [2,2]

        sqrt_2beta_Gamma = torch.linalg.cholesky(2 * beta * Gamma)  # [2,2]

        for i in range(self.n_steps - 1):
            z = zs[:, i, :]
            x, p, s = z[:, 0], z[:, 1], z[:, 2]

            z_vec = torch.stack([x, p, s], dim=1)
            A_z = torch.stack([
                p,
                -x - gamma**2 * p - gamma * lmbda * c * s,
                -gamma * lmbda * c * p - lmbda**2 * s
            ], dim=1)

            score = self._score_fn(z_vec, i)  # shape: [batch_size, 2]
            aux_force = (2 * beta) * (score @ Gamma.T)  # shape: [batch_size, 2]

            # Sample noise
            dW = torch.randn(z.shape[0], 2, device=DEVICE) * torch.sqrt(self.dt)
            noise = (dW @ sqrt_2beta_Gamma.T)  # shape: [batch_size, 2]

            dz = beta * A_z * self.dt
            dz[:, 1:] += aux_force * self.dt + noise

            zs[:, i+1, :] = zs[:, i, :] + dz

        return zs

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

    def run_demonstration(self, n_plot, n_hist):
        print(f"Running demonstration for {self.name}...")
        x0 = self._get_initial_samples(n_plot)
        z0_vars = [torch.randn(n_plot, device=DEVICE) * np.sqrt(s) for s in self.s_inits]
        z0 = torch.stack([x0] + z0_vars, dim=-1)
        xT_hist = torch.randn(n_hist, device=DEVICE)
        pT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        sT_hist = torch.randn(n_hist, device=DEVICE)
        zT_hist = torch.stack([xT_hist, pT_hist, sT_hist], dim=1)
        forward_paths = self.solve_forward_sde(z0).cpu().numpy()
        reverse_paths = self.solve_reverse_sde(zT_hist).cpu().numpy() # np.zeros_like(forward_paths)

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