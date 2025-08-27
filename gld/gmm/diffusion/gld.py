import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel
import scipy.linalg
from matrix_exp import stationary_covariance, compute_mean_and_covariance

DEVICE = torch.device("cpu")

class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Implements the forward and reverse processes for a generalized Langevin diffusion SDE.
    The state vector is z = [x, p, s], representing position, momentum, and an auxiliary variable.
    Forward SDE: dz = -beta * A * z * dt + G * dW
    """
    def __init__(self, gmm_params, **kwargs):
        super().__init__('Generalized Langevin Diffusion', gmm_params, **kwargs)
        # --- Model Parameters ---
        self.gamma = 2.
        self.c = 0.1
        self.lambda_val = 2/(1-self.c**2)
        self.M = 1.
        self.M_inv = 1. / self.M
        self.beta = 8. * np.sqrt(self.M)

        self.p_init_var = 0.01 * self.M
        self.s_init_var = 0.04

        self.A = torch.tensor([
            [0., -self.M_inv, 0.],
            [1., self.M_inv * self.gamma**2, self.gamma * self.lambda_val * self.c],
            [0., self.gamma * self.lambda_val * self.c, self.lambda_val**2]
        ], dtype=torch.float32, device=DEVICE)

        self.B = torch.tensor([
            [0., 0., 0.],
            [0., self.gamma, 0.],
            [0., self.lambda_val * self.c, self.lambda_val * np.sqrt(1 - self.c**2)]
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
        C_np = stationary_covariance(self.beta, A_np, G_np)

        for t_idx, t in enumerate(ts_np):
            means_k = []
            covs_k = []
            for k in range(n_components):
                mu0_k_np = np.array([self.gmm_params['means'][k], 0, 0])
                Sigma0_k_np = np.diag([
                    self.gmm_params['stds'][k]**2, self.p_init_var, self.s_init_var
                ])

                mu_t_np, Sigma_t_np = compute_mean_and_covariance(
                    t, self.beta, A_np, G_np, mu0_k_np, Sigma0_k_np, C_np
                )
                mu_t = torch.from_numpy(mu_t_np).float().to(DEVICE)
                Sigma_t = torch.from_numpy(Sigma_t_np).float().to(DEVICE)
                means_k.append(mu_t)
                covs_k.append(Sigma_t)
            # print('t_idx', t_idx)
            # print('mean', means_k)
            # print('covs', covs_k)
            self.perturbation_cache[t_idx] = {
                'weights': self.gmm_params['weights'],
                'means': means_k,
                'covs': covs_k,
            }
        print("Precomputation finished.")

    def solve_forward_sde_em(self, z0):
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
    
    def solve_forward_sde_anal(self, z0):
        """Solves the forward SDE using precomputed perturbation kernels (cached GMM)."""
        print(f"Solving forward SDE analytically for {self.name}...")
        zs = torch.zeros((z0.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, 0, :] = z0
        for i in range(1, self.n_steps):
            weights, means_k, covs_k = self._get_perturbed_params(i)
            component_probs = torch.tensor(weights, device=DEVICE)
            component_indices = torch.multinomial(component_probs, z0.shape[0], replacement=True)
            z_samples = []
            for j, comp_idx in enumerate(component_indices):
                mean_k = means_k[comp_idx]
                cov_k = covs_k[comp_idx]
                stable_cov = cov_k + 1e-6 * torch.eye(3, device=DEVICE)
                dist = torch.distributions.MultivariateNormal(mean_k, stable_cov)
                z_samples.append(dist.sample())
            zs[:, i, :] = torch.stack(z_samples)
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
    

    def precompute_sscs(self):
        """
        Precomputes everything needed for the analytical score and the SSCS sampler.
        """
        print("Precomputing for SSCS sampler...")
        A_H = np.array([
            [0., -self.M_inv, 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        A_O = self.A.cpu().numpy() - A_H
        
        F_O = -self.beta * A_O
        dt_half = (self.dt / 2).item()
        
        exp_FO_half_np = scipy.linalg.expm(-F_O * dt_half)
        self.exp_FO_half = torch.from_numpy(exp_FO_half_np).float().to(DEVICE)

        C_O = stationary_covariance(self.beta, A_O, self.G.cpu().numpy())
        _, Sigma_OU_half_np = compute_mean_and_covariance(dt_half, self.beta, A_O, self.G.cpu().numpy(), mu_0=np.zeros(3), Sigma_0=np.zeros((3, 3)), C=C_O)
        self.L_OU_half = torch.from_numpy(np.linalg.cholesky(Sigma_OU_half_np + 1e-9 * np.eye(3))).float().to(DEVICE)
        print("SSCS precomputation finished.")
    
    def solve_reverse_sde_sscs(self, zT):
        """
        Solves the reverse SDE using a symmetric splitting integrator:
        A_H(dt/2) -> A_O(dt/2) -> S(dt) -> A_O(dt/2) -> A_H(dt/2)
        """
        print(f"Solving reverse SDE for {self.name} with SSCS...")
        zs = torch.zeros_like(zT).unsqueeze(1).repeat(1, self.n_steps, 1)
        zs[:, -1, :] = zT
        
        dt_half = self.dt / 2.0

        for i in range(self.n_steps - 1, 0, -1):
            z = zs[:, i, :]
            t_idx = i

            # --- Step 1: Hamiltonian Half-Step (A_H for dt/2) ---
            x, p, s = z.split(1, dim=-1)
            p1 = p + (self.beta * x) * dt_half
            x1 = x - (self.beta * self.M_inv * p1) * dt_half
            z1 = torch.cat([x1, p1, s], dim=-1)

            # --- Step 2: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            # Apply pre-computed drift and diffusion
            z2_drift = (self.exp_FO_half @ z1.T).T
            noise1 = (self.L_OU_half @ torch.randn_like(z1).T).T
            z2 = z2_drift + noise1

            # --- Step 3: Score Full-Step (S for dt) ---
            score_old = self._score_fn(z2, t_idx)
            score_drift_old = (self.GGt @ score_old.T).T
            z3 = z2 + score_drift_old * self.dt

            # --- Step 4: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            z4_drift = (self.exp_FO_half @ z3.T).T
            noise2 = (self.L_OU_half @ torch.randn_like(z3).T).T
            z4 = z4_drift + noise2

            # --- Step 5: Hamiltonian Half-Step (A_H for dt/2, symmetric) ---
            x4, p4, s4 = z4.split(1, dim=-1)
            x5 = x4 - (self.beta * self.M_inv * p4) * dt_half
            p5 = p4 + (self.beta * x5) * dt_half
            z_next = torch.cat([x5, p5, s4], dim=-1)
            
            zs[:, i-1, :] = z_next

        return zs


    def solve_reverse_sde_ubu(self, zT):
        return None
    
    def solve_reverse_sde(self, zT, type='em'):
        if type=='em':
            return self.solve_reverse_sde_em(zT)
        elif type == 'sscs':
            self.precompute_sscs()
            return self.solve_reverse_sde_sscs(zT)
        elif type == 'ubu':
            return self.solve_reverse_sde_ubu(zT)
        
    def solve_forward_sde(self, zT, type='em'):
        if type=='em':
            return self.solve_forward_sde_em(zT)
        elif type == 'sscs':
            self.precompute_sscs()
            return self.solve_forward_sde_sscs(zT)
        elif type == 'ubu':
            return self.solve_forward_sde_ubu(zT)
        elif type == 'anal':
            return self.solve_forward_sde_anal(zT)

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

        forward_sde_paths = self.solve_forward_sde(z0, type='em').cpu().numpy()

        xT_hist = torch.randn(n_hist, device=DEVICE)
        pT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        sT_hist = torch.randn(n_hist, device=DEVICE)
        zT_hist = torch.stack([xT_hist, pT_hist, sT_hist], dim=1)
        reverse_sde_paths = self.solve_reverse_sde(zT_hist, type='sscs').cpu().numpy()
        #reverse_ode_paths = self.solve_pfode(zT_hist).cpu().numpy()

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        var_names = ['Position (x)', 'Momentum (p)', 'Memory (s)']
        ts_cpu = self.ts.cpu().numpy()
        
        for i in range(3):
            # Forward
            axes[i, 0].plot(ts_cpu, forward_sde_paths[:, :, i].T, lw=1.5, alpha=0.6)
            axes[i, 0].set_title(f'Forward: {var_names[i]}')
            axes[i, 0].set_ylabel(var_names[i])
            axes[i, 0].set_ylim(-6, 6)

            # Reverse
            axes[i, 1].plot(ts_cpu, reverse_sde_paths[:n_plot, :, i].T, lw=1.5, alpha=0.5)
            #axes[i, 1].plot(ts_cpu, reverse_ode_paths[:n_plot, :, i].T, lw=1.0, alpha=0.8, color='green')
            axes[i, 1].set_title(f'Reverse: {var_names[i]}')
            axes[i, 1].set_ylim(-6, 6)

            if i == 2:
                axes[i, 0].set_xlabel('Time')
                axes[i, 1].set_xlabel('Time')
                axes[i, 2].set_xlabel('Value')

        position_data = reverse_sde_paths[:, 0, 0]
        momentum_data = reverse_sde_paths[:, 0, 1]
        memory_data = reverse_sde_paths[:, 0, 2]

        # Remove outliers > 1000
        position_filtered = position_data[np.abs(position_data) <= 1000]
        momentum_filtered = momentum_data[np.abs(momentum_data) <= 1000]
        memory_filtered = memory_data[np.abs(memory_data) <= 1000]

        # Final distributions (3rd column, set x-limits)
        plot_position_dist(position_filtered, self.gmm_params, axes[0, 2])
        axes[0, 2].set_title("Final Position Distribution")
        axes[0, 2].set_xlim(-6, 6)

        plot_aux_dist(axes[1, 2], (momentum_filtered, 'Momentum'),
                    target_dist=(0, np.sqrt(self.p_init_var)))
        axes[1, 2].set_title("Final Momentum Distribution")
        axes[1, 2].set_xlim(-4, 4)

        plot_aux_dist(axes[2, 2], (memory_filtered, 'Memory'),
                    target_dist=(0, np.sqrt(self.s_init_var)))
        axes[2, 2].set_title("Final Memory Distribution")
        axes[2, 2].set_xlim(-4, 4)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
