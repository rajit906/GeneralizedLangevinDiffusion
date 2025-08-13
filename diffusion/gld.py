# TODO: 
# Fix numerical stability with lam=0. 
# Cache the covariance integration in matrix_exp if possible. 
# Implement PFODE with inbuilt solver.
import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel
from matrix_exp import compute_mean_and_covariance
import scipy.linalg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Implements the forward and reverse processes for a generalized Langevin diffusion SDE.
    The state vector is z = [x, p, s], representing position, momentum, and an auxiliary variable.
    Forward SDE: dz = -beta * A * z * dt + G * dW
    """
    def __init__(self, gmm_params, **kwargs):
        super().__init__('Generalized Langevin Diffusion', gmm_params, **kwargs)
        # --- Model Parameters ---
        self.beta = 4.0
        self.gamma = 1.0
        self.lambda_val = 1.
        self.c = 1.
        self.M = 0.25
        self.M_inv = 1.0 / self.M

        self.p_init_var = self.gamma * self.M
        self.s_init_var = 0.01

        self.A = torch.tensor([
            [0, -self.M_inv, 0],
            [1, self.M_inv * self.gamma**2, self.gamma * self.lambda_val * self.c],
            [0, self.gamma * self.lambda_val * self.c, self.lambda_val**2]
        ], dtype=torch.float32, device=DEVICE)

        self.B = torch.tensor([
            [0, 0, 0],
            [0, self.gamma, 0],
            [0, self.gamma * self.lambda_val * self.c, self.lambda_val * np.sqrt(1 - self.c**2)]
        ], dtype=torch.float32, device=DEVICE)
        self.G = np.sqrt(2 * self.beta) * self.B
        self.GGt = self.G @ self.G.T
        self.perturbation_cache = {}
        self.precompute()

    def precompute(self):
        """
        Precomputes the evolution of each GMM component's mean and covariance over time
        by calling the accurate analytical solver from matrix_exp.py.
        """
        print("Precomputing perturbation kernels analytically...")
        n_components = len(self.gmm_params['weights'])
        A_np = self.A.cpu().numpy()
        G_np = self.G.cpu().numpy()
        ts_np = self.ts.cpu().numpy()

        for t_idx, t in enumerate(ts_np):
            means_k = []
            covs_k = []
            for k in range(n_components):
                mu0_k_np = np.array([self.gmm_params['means'][k], 0, 0])
                Sigma0_k_np = np.diag([
                    self.gmm_params['stds'][k]**2, self.p_init_var, self.s_init_var
                ])

                mu_t_np, Sigma_t_np = compute_mean_and_covariance(
                    t, self.beta, A_np, G_np, mu0_k_np, Sigma0_k_np
                )
                mu_t = torch.from_numpy(mu_t_np).float().to(DEVICE)
                Sigma_t = torch.from_numpy(Sigma_t_np).float().to(DEVICE)
                means_k.append(mu_t)
                covs_k.append(Sigma_t)

            self.perturbation_cache[t_idx] = {
                'weights': self.gmm_params['weights'],
                'means': means_k,
                'covs': covs_k,
            }
        print("Precomputation finished.")

    def precompute_sscs_params(self):
        """Precomputes matrices needed for the SSCS solver."""
        print("Precomputing matrices for SSCS solver...")
        h = self.dt.item()
        A_np = self.A.cpu().numpy()
        G_np = self.G.cpu().numpy()
        # Propagator for the linear drift part over half a time step
        # Corresponds to solving dZ/dt = -beta * A * Z
        M_h_half_np = scipy.linalg.expm(- (h / 2.0) * self.beta * A_np)
        # Covariance of the noise for the linear SDE part over half a time step
        # This is the solution to the Lyapunov equation, obtained by calling the
        # analytical solver with zero initial mean and covariance.
        _, Sigma_h_half_np = compute_mean_and_covariance(
            h / 2.0, self.beta, A_np, G_np, np.zeros(3), np.zeros((3, 3))
        )
        # Cholesky decomposition for efficient noise sampling
        # Add a small identity matrix for numerical stability before decomposition
        L_h_half_np = np.linalg.cholesky(Sigma_h_half_np + 1e-8 * np.eye(3))
        # Store matrices as torch tensors on the correct device
        self.M_h_half = torch.from_numpy(M_h_half_np).float().to(DEVICE)
        self.L_h_half = torch.from_numpy(L_h_half_np).float().to(DEVICE)
        print("SSCS precomputation finished.")


    def solve_forward_sde(self, z0):
        """Solves the forward SDE dz = -beta * A * z * dt + G * dW using Euler-Maruyama."""
        print(f"Solving forward SDE for {self.name}...")
        zs = torch.zeros((z0.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, 0, :] = z0
        sqrt_dt = torch.sqrt(self.dt)

        for i in range(self.n_steps - 1):
            z = zs[:, i, :]
            dW = torch.randn_like(z) * sqrt_dt
            drift = -self.beta * (self.A @ z.T).T
            diffusion = (self.G @ dW.T).T
            dz = drift * self.dt + diffusion
            zs[:, i + 1, :] = z + dz
        return zs

    def _get_perturbed_params(self, t_idx):
        """
        Retrieves the precomputed GMM parameters (weights, means, covs) for a given time step.
        """
        cached_params = self.perturbation_cache[t_idx]
        return cached_params['weights'], cached_params['means'], cached_params['covs']
    def _score_fn(self, z, t_idx):
        """
        Computes the marginal score \nabla_z log p_t(z) for the GMM analytically.
        """
        weights, means_k, covs_k = self._get_perturbed_params(t_idx)
        batch_size = z.shape[0]
        p_t_z = torch.zeros(batch_size, device=DEVICE)
        grad_v_p_t_z = torch.zeros(batch_size, 3, device=DEVICE)
        for w, mean, cov in zip(weights, means_k, covs_k):
            stable_cov = cov + 1e-6 * torch.eye(3, device=DEVICE)
            dist_3d = torch.distributions.MultivariateNormal(mean, stable_cov)
            pdf = torch.exp(dist_3d.log_prob(z))
            score = -torch.linalg.solve(stable_cov, (z - mean).T).T
            p_t_z += w * pdf
            grad_v_p_t_z += (w * pdf).unsqueeze(1) * score
        final_score_3d = grad_v_p_t_z / (p_t_z.unsqueeze(1) + 1e-8)
        return final_score_3d

    def solve_reverse_sde_em(self, zT):
        """
        Solves the reverse SDE dz = [-beta*A*z - G*G^T*S']dt + G*dW_bar using Euler-Maruyama.
        """
        print(f"Solving reverse SDE for {self.name}...")
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
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
                zs[:, i - 1, :] = z - drift_rev * self.dt + diffusion
        return zs
    
    def solve_reverse_sde_sscs(self, zT):
        """
        Solves the reverse SDE using a symmetric splitting scheme (A-B-A).
        - A-step: Exact solution of the linear OU process part.
        - B-step: Euler step for the non-linear score drift part.
        """
        print(f"Solving reverse SDE for {self.name} with SSCS...")
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, -1, :] = zT
        h = self.dt

        # Iterate backwards from T to 0
        for i in range(self.n_steps - 1, 0, -1):
            z_curr = zs[:, i, :]
            # --- First A-Step: Evolve by -h/2 with the linear OU process ---
            mu1 = z_curr @ self.M_h_half.T
            noise1 = torch.randn_like(z_curr) @ self.L_h_half.T
            z_half = mu1 + noise1
            # --- B-Step: Evolve by -h with the score drift (Euler step) ---
            # The state z_half is at the midpoint in time, t_i - h/2.
            # We approximate the score at this time by using the parameters for t_i.
            score_full = self._score_fn(z_half, i)
            score_drift = (self.GGt @ score_full.T).T
            # The reverse drift for this part is `score_drift`. We evolve backwards by h.
            z_half_prime = z_half - h * score_drift
            # --- Second A-Step: Evolve by -h/2 with the linear OU process ---
            mu2 = z_half_prime @ self.M_h_half.T
            noise2 = torch.randn_like(z_curr) @ self.L_h_half.T
            z_next = mu2 + noise2
            zs[:, i - 1, :] = z_next
        return zs
    
    def solve_reverse_sde_ubu(self, zT):
        return None
    
    def solve_reverse_sde(self, zT, type='em'):
        if type=='em':
            return self.solve_reverse_sde_em(zT)
        elif type == 'sscs':
            self.precompute_sscs_params()
            return self.solve_reverse_sde_sscs(zT)
        elif type == 'ubu':
            return self.solve_reverse_sde_ubu(zT)

    def solve_pfode(self, zT):
        """
        Solves the reverse process using the Probability Flow ODE (deterministic).
        dz = [-f_fwd(z) + 0.5 * G*G^T*S']dt
        """
        print(f"Solving reverse PF-ODE for {self.name}...")
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, -1, :] = zT

        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]
            score_full = self._score_fn(z, i)
            f_fwd = -self.beta * (self.A @ z.T).T
            score_drift = (self.GGt @ score_full.T).T
            drift_ode = -f_fwd + 0.5 * score_drift
            if i > 0:
                zs[:, i - 1, :] = z - drift_ode * self.dt
        return zs

    def run_demonstration(self, n_plot, n_hist):
        """
        Runs and visualizes both the forward and reverse SDE/ODE processes.
        """
        print(f"\nRunning demonstration for {self.name}...")
        x0 = self._get_initial_samples(n_plot)
        p0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.p_init_var)
        s0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.s_init_var)
        z0 = torch.stack([x0, p0, s0], dim=-1)

        forward_sde_paths = self.solve_forward_sde(z0).cpu().numpy()

        xT_hist = torch.randn(n_hist, device=DEVICE)
        pT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        sT_hist = torch.randn(n_hist, device=DEVICE)
        zT_hist = torch.stack([xT_hist, pT_hist, sT_hist], dim=1)
        reverse_sde_paths = self.solve_reverse_sde(zT_hist, type='sscs').cpu().numpy()
        reverse_ode_paths = self.solve_pfode(zT_hist).cpu().numpy()

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        var_names = ['Position (x)', 'Momentum (p)', 'Memory (s)']
        ts_cpu = self.ts.cpu().numpy()
        for i in range(3):
            axes[i, 0].plot(ts_cpu, forward_sde_paths[:, :, i].T, lw=1.5, alpha=0.6, color='blue')
            axes[i, 0].set_title(f'Forward: {var_names[i]}')
            axes[i, 0].set_ylabel(var_names[i])
            axes[i, 1].plot(ts_cpu, reverse_sde_paths[:n_plot, :, i].T, lw=1.5, alpha=0.5)
            #axes[i, 1].plot(ts_cpu, reverse_ode_paths[:n_plot, :, i].T, lw=1.0, alpha=0.8, color='green')
            axes[i, 1].set_title(f'Reverse: {var_names[i]}')
            if i == 2:
                axes[i, 0].set_xlabel('Time')
                axes[i, 1].set_xlabel('Time')
                axes[i, 2].set_xlabel('Value')

        plot_position_dist(reverse_sde_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        axes[0, 2].hist(reverse_ode_paths[:, 0, 0], bins=50, density=True, alpha=0.6, color='green')
        axes[0, 2].set_title("Final Position Distribution")

        plot_aux_dist(axes[1, 2], (reverse_sde_paths[:, 0, 1], 'Momentum'), target_dist=(0, np.sqrt(self.p_init_var)))
        axes[1, 2].hist(reverse_ode_paths[:, 0, 1], bins=50, density=True, alpha=0.6, color='green')
        axes[1, 2].set_title("Final Momentum Distribution")

        plot_aux_dist(axes[2, 2], (reverse_sde_paths[:, 0, 2], 'Memory'), target_dist=(0, np.sqrt(self.s_init_var)))
        axes[2, 2].hist(reverse_ode_paths[:, 0, 2], bins=50, density=True, alpha=0.6, color='green')
        axes[2, 2].set_title("Final Memory Distribution")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()