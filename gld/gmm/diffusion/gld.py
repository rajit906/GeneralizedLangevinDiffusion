import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel
import scipy.linalg
from matrix_exp import stationary_covariance, compute_mean_and_covariance
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cpu")

class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Implements the forward and reverse processes for a generalized Langevin diffusion SDE.
    The state vector is z = [x, p, s], representing position, momentum, and an auxiliary variable.
    Forward SDE: dz = -beta * A * z * dt + G * dW
    """
    def __init__(self, gmm_params, inttype= 'em', gamma=1.0,lambda_val=1.0, c_val = 0.1, M=1., **kwargs):
        super().__init__('Generalized Langevin Diffusion', gmm_params, **kwargs)
        # --- Model Parameters ---
        self.inttype = inttype
        self.gamma = gamma
        self.c = c_val
        self.lambda_val = lambda_val
        self.M = M
        self.M_inv = 1. / self.M
        self.beta = 8. * np.sqrt(self.M)

        self.p_init_var = 1.#0.01 * self.M
        self.s_init_var = 1.#0.04

        self.AH = torch.tensor([
            [0., -self.M_inv, 0.],
            [1., 0.,0.],
            [0., 0.,0.]
        ], dtype=torch.float32, device=DEVICE)
        
        self.AGamma = torch.tensor([
            [0., 0., 0.],
            [0., self.M_inv * self.gamma**2, self.gamma * self.lambda_val * self.c],
            [0., self.gamma * self.lambda_val * self.c, self.lambda_val**2]
        ], dtype=torch.float32, device=DEVICE)
        
        self.A = self.AH + self.AGamma

        self.B = torch.tensor([
            [0., 0., 0.],
            [0., self.gamma, 0.],
            [0., self.lambda_val * self.c, self.lambda_val * np.sqrt(1 - self.c**2)]
        ], dtype=torch.float32, device=DEVICE)

        self.G = np.sqrt(2 * self.beta) * self.B
        self.GGt = self.G @ self.G.T
        self.A_np = self.A.cpu().numpy()
        self.G_np = self.G.cpu().numpy()
        self.C_np = stationary_covariance(self.beta, self.A_np, self.G_np)
        
        
        dt = self.dt.item()
        self.AH_np = self.AH.cpu().numpy()
        self.AGamma_np = self.AGamma.cpu().numpy()
        self.F_half_np = scipy.linalg.expm( self.beta * (self.AH_np-self.AGamma_np) * dt/2)
        self.Ft_half_np = scipy.linalg.expm(self.beta * (2*self.AGamma_np) * dt/2)
        #print("F_half_np = ", self.F_half_np)
        self.S_half_np = np.linalg.cholesky(np.eye(3) - self.F_half_np @ self.F_half_np.T)
        #self.St_half_np = np.linalg.cholesky(np.eye(3) - self.Ft_half_np @ self.Ft_half_np.T)
        #self.L_half_np = np.linalg.cholesky(Sigma_np + 1e-9 * np.eye(3))

        # self.F_half = torch.from_numpy(F_half_np).float().to(DEVICE)
        # self.L_half = torch.from_numpy(L_half_np).float().to(DEVICE)
        

    def precompute(self):
        """Method is required by the abstract base class, but we do nothing here."""
        pass

    def solve_forward_sde_em(self, z0):
        """Solves the forward SDE dz = -beta * A * z * dt + G * dW using Euler-Maruyama."""
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
        """
        Solves the forward SDE analytically by propagating sample paths
        with the exact Gaussian transition kernel:
            z_{t+dt} = F z_t + L ξ,   ξ ~ N(0, I)
        """
        batch_size = z0.shape[0]
        zs = torch.zeros((batch_size, self.n_steps, 3), device=DEVICE)
        zs[:, 0, :] = z0

        # --- Precompute exact transition kernel for one step ---
        A_np = self.A.cpu().numpy()
        G_np = self.G.cpu().numpy()
        dt = self.dt.item()
        F_np = scipy.linalg.expm(-self.beta * A_np * dt)
        _, Sigma_np = compute_mean_and_covariance(
            dt, self.beta, A_np, G_np,
            mu_0=np.zeros(3), Sigma_0=np.zeros((3, 3)),
            C=stationary_covariance(self.beta, A_np, G_np)
        )
        L_np = np.linalg.cholesky(Sigma_np + 1e-9 * np.eye(3))

        F = torch.from_numpy(F_np).float().to(DEVICE)
        L = torch.from_numpy(L_np).float().to(DEVICE)

        # --- Propagate paths ---
        for i in range(self.n_steps - 1):
            z = zs[:, i, :]
            noise = (L @ torch.randn(batch_size, 3, device=DEVICE).T).T
            zs[:, i+1, :] = (F @ z.T).T + noise

        return zs



    def _get_perturbed_params(self, t):
        """
        Computes the GMM parameters (weights, means, covs) for a given time t.
        """
        n_components = len(self.gmm_params['weights'])
        means_k = []
        covs_k = []

        for k in range(n_components):
            mu0_k_np = np.array([self.gmm_params['means'][k], 0, 0])
            Sigma0_k_np = np.diag([
                self.gmm_params['stds'][k]**2, self.p_init_var, self.s_init_var
            ])

            mu_t_np, Sigma_t_np = compute_mean_and_covariance(
                t, self.beta, self.A_np, self.G_np, mu0_k_np, Sigma0_k_np, self.C_np
            )
            mu_t = torch.from_numpy(mu_t_np).float().to(DEVICE)
            Sigma_t = torch.from_numpy(Sigma_t_np).float().to(DEVICE)
            means_k.append(mu_t)
            covs_k.append(Sigma_t)
            
        return self.gmm_params['weights'], means_k, covs_k
    

    def _score_fn(self, z, t_idx):
        """
        Computes the marginal score \nabla_z log p_t(z) for the GMM analytically
        in a vectorized fashion, without using MultivariateNormal or linalg.solve.
        """
        t = self.ts[t_idx].item()
        # _get_perturbed_params returns the weights tensor, and lists of mean and cov tensors
        weights, means_k, covs_k = self._get_perturbed_params(t)
        means = torch.stack(means_k) # Shape: (K, 3)
        covs = torch.stack(covs_k)   # Shape: (K, 3, 3)
        N, d = z.shape[0], z.shape[1]
        stable_covs = covs + 1e-6 * torch.eye(d, device=DEVICE).unsqueeze(0)
        cov_invs = torch.linalg.inv(stable_covs) # Shape: (K, 3, 3)
        log_dets = torch.linalg.slogdet(stable_covs)[1] # Shape: (K,)
        z_expanded = z.unsqueeze(1) # Shape: (N, 1, 3)
        diff = z_expanded - means # Shape: (N, K, 3) via broadcasting
        mat_vec_prod = torch.matmul(cov_invs, diff.unsqueeze(-1)).squeeze(-1)
        mahalanobis_term = torch.sum(diff * mat_vec_prod, dim=-1) # Shape: (N, K)
        log_2pi = d * np.log(2 * np.pi)
        log_pdfs = -0.5 * (log_2pi + log_dets + mahalanobis_term) # Shape: (N, K)
        pdfs = torch.exp(log_pdfs)
        weighted_pdfs = pdfs * weights # Shape: (N, K)
        p_t_z = torch.sum(weighted_pdfs, dim=1) # Shape: (N,)
        per_component_scores = -mat_vec_prod # Shape: (N, K, 3)
        grad_v_p_t_z = torch.sum(weighted_pdfs.unsqueeze(-1) * per_component_scores, dim=1) # Shape: (N, 3)
        final_score_3d = grad_v_p_t_z / (p_t_z.unsqueeze(1) + 1e-8)
        return final_score_3d

    def solve_reverse_sde_em(self, zT, score_model=None):
        """
        Solves the reverse SDE dz = [-beta*A*z - G*G^T*S']dt + G*dW_bar using Euler-Maruyama.
        """
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, -1, :] = zT
        sqrt_dt = torch.sqrt(self.dt)
        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]
            if score_model is None:
                score_full = self._score_fn(z, i)  # (B,3)
            else:
                t_idx = torch.full((z.shape[0],), i, device=DEVICE, dtype=torch.long)
                ps = z[:, 1:]  # (B,2)
                score_ps = score_model(ps, t_idx)  # (B,2)
                score_full = torch.cat([torch.zeros_like(ps[:, :1]), score_ps], dim=1)  # (B,3)
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
        A_H = np.array([
            [0., -self.M_inv, 0.],
            [1., 0., 0.],
            [0., 0., 0.]
        ])
        A_O = self.A.cpu().numpy() - A_H
        dt_half = (self.dt / 2).item()
        
        F_O = -self.beta * A_O
        self.F_O = F_O
        exp_FO_half_np = scipy.linalg.expm(F_O * dt_half)
        self.exp_FO_half = torch.from_numpy(exp_FO_half_np).float().to(DEVICE)
        C_O = stationary_covariance(self.beta, A_O, self.G.cpu().numpy())
        _, Sigma_OU_half_np = compute_mean_and_covariance(dt_half, self.beta, A_O, self.G.cpu().numpy(), mu_0=np.zeros(3), Sigma_0=np.zeros((3, 3)), C=C_O)
        self.L_OU_half = torch.from_numpy(np.linalg.cholesky(Sigma_OU_half_np + 1e-9 * np.eye(3))).float().to(DEVICE)
    
    def solve_reverse_sde_sscs(self, zT, score_model=None):
        """
        Solves the reverse SDE using a symmetric splitting integrator:
        A_H(dt/2) -> A_O(dt/2) -> S(dt) -> A_O(dt/2) -> A_H(dt/2)
        """
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

            # --- Step 3: Score Full-Euler-Step (S for dt) ---
            if score_model is None:
                score_old = self._score_fn(z2, t_idx)
            else:
                t_idx_tensor = torch.full((z2.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
                ps = z2[:, 1:]
                score_ps = score_model(ps, t_idx_tensor)
                score_old = torch.cat([torch.zeros_like(ps[:, :1]), score_ps], dim=1)
            score_drift_old = (self.GGt @ (score_old.T + z2.T)).T
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
    
    def solve_forward_sde_sscs(self, z0):
        """
        Solves the forward SDE using a symmetric splitting integrator
        (no score step needed in the forward direction):
        A_H(dt/2) -> A_O(dt) -> A_H(dt/2)

        Since we only have half-step operators, we apply them twice
        to achieve the full Ornstein–Uhlenbeck step.
        """
        zs = torch.zeros_like(z0).unsqueeze(1).repeat(1, self.n_steps, 1)
        zs[:, 0, :] = z0
        
        dt_half = self.dt / 2.0

        for i in range(0, self.n_steps - 1):
            z = zs[:, i, :]

            # --- Step 1: Hamiltonian Half-Step (A_H for dt/2) ---
            x, p, s = z.split(1, dim=-1)
            p1 = p + (self.beta * x) * dt_half
            x1 = x - (self.beta * self.M_inv * p1) * dt_half
            z1 = torch.cat([x1, p1, s], dim=-1)

            # --- Step 2: Ornstein-Uhlenbeck Full-Step (A_O for dt) ---
            # Implemented as two half-steps
            # Half-step 1
            z2_drift = (self.exp_FO_half @ z1.T).T
            noise1 = (self.L_OU_half @ torch.randn_like(z1).T).T
            z2 = z2_drift + noise1
            # Half-step 2
            z3_drift = (self.exp_FO_half @ z2.T).T
            noise2 = (self.L_OU_half @ torch.randn_like(z2).T).T
            z3 = z3_drift + noise2

            # --- Step 3: Hamiltonian Half-Step (A_H for dt/2) ---
            x3, p3, s3 = z3.split(1, dim=-1)
            x4 = x3 - (self.beta * self.M_inv * p3) * dt_half
            p4 = p3 + (self.beta * x4) * dt_half
            z_next = torch.cat([x4, p4, s3], dim=-1)

            zs[:, i+1, :] = z_next

        return zs
    
    def precompute_sscs2(self):

        self.F_half = torch.from_numpy(self.F_half_np).float().to(DEVICE)
        self.S_half = torch.from_numpy(self.S_half_np).float().to(DEVICE)
   
    def solve_reverse_sde_sscs2(self, zT, score_model=None):
        """
        Solves the reverse SDE using a symmetric splitting integrator:
        A_H(dt/2) -> A_O(dt/2) -> S(dt) -> A_O(dt/2) -> A_H(dt/2)
        """
        zs = torch.zeros_like(zT).unsqueeze(1).repeat(1, self.n_steps, 1)
        zs[:, -1, :] = zT
        
        dt_half = self.dt / 2.0

        for i in range(self.n_steps - 1, 0, -1):
            z1 = zs[:, i, :]
            t_idx = i

            # --- Step 2: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            # Apply pre-computed drift and diffusion
            z2_drift = (self.F_half  @ z1.T).T
            noise1 = (self.S_half @ torch.randn_like(z1).T).T
            z2 = z2_drift + noise1

            # # --- Step 3: Score Full-Euler-Step (S for dt) ---
            if score_model is None:
                score_old = self._score_fn(z2, t_idx)
            else:
                t_idx_tensor = torch.full((z2.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
                ps = z2[:, 1:]
                score_ps = score_model(ps, t_idx_tensor)
                score_old = torch.cat([torch.zeros_like(ps[:, :1]), score_ps], dim=1)
            score_drift_old = (self.GGt @ (score_old.T + z2.T)).T
            z3 = z2 + score_drift_old * self.dt
            # --- Step 4: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            z4_drift = (self.F_half @ z3.T).T
            noise2 = (self.S_half @ torch.randn_like(z3).T).T
            z4 = z4_drift + noise2
            
            zs[:, i-1, :] = z4

        return zs
    
    def precompute_sscs3(self):
        self.F_half = torch.from_numpy(self.F_half_np).float().to(DEVICE)
        self.S_half = torch.from_numpy(self.S_half_np).float().to(DEVICE)
        self.Ft_half = torch.from_numpy(self.Ft_half_np).float().to(DEVICE)
   
    def solve_reverse_sde_sscs3(self, zT, score_model=None):
        """
        Solves the reverse SDE using a symmetric splitting integrator:
        (At_H(dt/2) + At_O(dt/2)) -> S(dt) -> (At_O(dt/2) + At_H(dt/2))
        """
        zs = torch.zeros_like(zT).unsqueeze(1).repeat(1, self.n_steps, 1)
        zs[:, -1, :] = zT
        
        dt_half = self.dt / 2.0

        for i in range(self.n_steps - 1, 0, -1):
            z1 = zs[:, i, :]
            t_idx = i
            # --- Step 2: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            # Apply pre-computed drift and diffusion
            z2_drift = (self.F_half  @ z1.T).T
            noise1 = (self.S_half @ torch.randn_like(z1).T).T
            z2 = z2_drift + noise1

            z2t = (self.Ft_half @ z2.T).T
            
            # # --- Step 3: Score Full-Euler-Step (S for dt) ---
            if score_model is None:
                score_old = self._score_fn(z2, t_idx)
            else:
                t_idx_tensor = torch.full((z2.shape[0],), t_idx, device=DEVICE, dtype=torch.long)
                ps = z2[:, 1:]
                score_ps = score_model(ps, t_idx_tensor)
                score_old = torch.cat([torch.zeros_like(ps[:, :1]), score_ps], dim=1)
            score_drift_old = (self.GGt @ score_old.T).T
            z3 = z2t + score_drift_old * self.dt
            
            z3t = (self.Ft_half @ z3.T).T
            # --- Step 4: Ornstein-Uhlenbeck Half-Step (A_O for dt/2) ---
            z4_drift = (self.F_half @ z3t.T).T
            noise2 = (self.S_half @ torch.randn_like(z3t).T).T
            z4 = z4_drift + noise2
            
            zs[:, i-1, :] = z4

        return zs
    
    def solve_reverse_sde_ubu(self, zT, score_model=None):
        return None
    
    def solve_reverse_sde(self, zT, type='em', score_model=None):
        if type=='em':
            return self.solve_reverse_sde_em(zT, score_model=score_model)
        elif type == 'sscs':
            self.precompute_sscs()
            return self.solve_reverse_sde_sscs(zT, score_model=score_model)
        elif type == 'sscs2':
            self.precompute_sscs2()
            return self.solve_reverse_sde_sscs2(zT, score_model=score_model)
        elif type == 'sscs3':
            self.precompute_sscs3()
            return self.solve_reverse_sde_sscs3(zT, score_model=score_model)
        elif type == 'ubu':
            return self.solve_reverse_sde_ubu(zT, score_model=score_model)
        
    def solve_forward_sde(self, z0, type='em'):
        if type=='em':
            return self.solve_forward_sde_em(z0)
        elif type == 'sscs':
            self.precompute_sscs()
            return self.solve_forward_sde_sscs(z0)
        elif type == 'ubu':
            return self.solve_forward_sde_ubu(z0)
        elif type == 'anal':
            return self.solve_forward_sde_anal(z0)

    def solve_pfode(self, zT):
        """
        Solves the reverse process using the Probability Flow ODE (deterministic).
        dz = [-f_fwd(z) + 0.5 * G*G^T*S']dt
        """
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

    def run_demonstration(self, n_plot, n_hist, score_model=None):
        """
        Runs and visualizes both the forward and reverse SDE/ODE processes,
        including histograms of terminal forward distributions vs. true law.
        """
        x0 = self._get_initial_samples(n_plot)
        p0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.p_init_var)
        s0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.s_init_var)
        z0 = torch.stack([x0, p0, s0], dim=-1)

        forward_sde_paths = self.solve_forward_sde(z0, type='sscs').cpu().numpy()

        xT_hist = torch.randn(n_hist, device=DEVICE)
        pT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        sT_hist = torch.randn(n_hist, device=DEVICE)
        zT_hist = torch.stack([xT_hist, pT_hist, sT_hist], dim=1)
        reverse_sde_paths = self.solve_reverse_sde(zT_hist, type=self.inttype, score_model=score_model).detach().cpu().numpy()

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        var_names = ['Position (x)', 'Momentum (p)', 'Memory (s)']
        ts_cpu = self.ts.cpu().numpy()

        for i in range(3):
            # Forward trajectories
            axes[i, 0].plot(ts_cpu, forward_sde_paths[10:, :, i].T, lw=1.5, alpha=0.05, color='darkblue')
            axes[i, 0].plot(ts_cpu, forward_sde_paths[:10, :, i].T, lw=1.5, alpha=1, color='darkblue')
            axes[i, 0].set_title(f'Forward: {var_names[i]}')
            axes[i, 0].set_ylabel(var_names[i])

            # Reverse trajectories
            axes[i, 1].plot(ts_cpu, reverse_sde_paths[10:n_plot, :, i].T, lw=1.5, alpha=0.05, color='darkblue')
            axes[i, 1].plot(ts_cpu, reverse_sde_paths[:10, :, i].T, lw=1.5, alpha=1, color='darkblue')
            axes[i, 1].set_title(f'Reverse: {var_names[i]}')

            if i == 2:
                axes[i, 0].set_xlabel('Time')
                axes[i, 1].set_xlabel('Time')
                axes[i, 2].set_xlabel('Value')
                #axes[i, 3].set_xlabel('Value')

        # === Reverse terminal distributions (col 3) ===
        position_data = reverse_sde_paths[:, 0, 0]
        momentum_data = reverse_sde_paths[:, 0, 1]
        memory_data  = reverse_sde_paths[:, 0, 2]

        position_filtered = position_data#[np.abs(position_data) <= 100]
        momentum_filtered = momentum_data#[np.abs(momentum_data) <= 100]
        memory_filtered   = memory_data#[np.abs(memory_data) <= 100]

        # Call plotting functions without the 'color' argument to fix the TypeError
        plot_position_dist(position_filtered, self.gmm_params, axes[0, 2])
        axes[0, 2].set_title("Final Position Distribution (Reverse)")

        plot_aux_dist(axes[1, 2], (momentum_filtered, 'Momentum'),
                    target_dist=(0, np.sqrt(self.p_init_var)))
        axes[1, 2].set_title("Final Momentum Distribution (Reverse)")

        plot_aux_dist(axes[2, 2], (memory_filtered, 'Memory'),
                    target_dist=(0, np.sqrt(self.s_init_var)))
        axes[2, 2].set_title("Final Memory Distribution (Reverse)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        return reverse_sde_paths

    def train_score_network_hsm(self, ScoreNetwork, n_epochs=50, batch_size=128, lr=1e-3, n_steps=1000):
        """
        Hybrid Score Matching with noise regression and GG^T-weighted loss.
        The model predicts the score on (p,s), we reparametrize to eps,
        and weight the regression loss with ||·||_{GG^T}. 
        TODO: Make sure perturbation kernel has correct initialization.
        """
        device = DEVICE
        model = ScoreNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        losses = []

        K = len(self.gmm_params['weights'])
        weights_tensor = torch.tensor(self.gmm_params['weights'], device=device, dtype=torch.float32)
        weights_tensor = weights_tensor / weights_tensor.sum()

        # Precompute GG^T
        if isinstance(self.GGt, torch.Tensor):
            GGt = self.GGt.float().to(device)
        else:
            GGt = torch.from_numpy(self.GGt).float().to(device)
        GGt_ps = GGt[1:, 1:]  # (2,2) block for (p,s)

        for epoch in range(n_epochs):
            total_loss = 0.0
            for step in range(n_steps):
                # --- 1. Sample x0 ---
                x0 = self._get_initial_samples(batch_size).to(device)  # (B,)

                # --- 2. Sample single time index ---
                t_idx_scalar = int(torch.randint(0, self.n_steps, (1,)).item())
                t = self.ts[t_idx_scalar].item()

                # --- 3. Perturbed params ---
                _, means_list, covs_list = self._get_perturbed_params(t)
                means_K = torch.stack(means_list, dim=0).to(device)   # (K,3)
                covs_K = torch.stack(covs_list, dim=0).to(device)     # (K,3,3)

                # --- 4. Mixture component choice ---
                comp_idx = torch.multinomial(weights_tensor, batch_size, replacement=True)
                mu_t_batch = means_K[comp_idx]   # (B,3)
                Sigma_t_batch = covs_K[comp_idx] # (B,3,3)

                # --- 5. Cholesky ---
                jitter = 1e-6
                L_t_batch = torch.linalg.cholesky(
                    Sigma_t_batch + jitter * torch.eye(3, device=device).unsqueeze(0)
                )  # (B,3,3)
                L_t_trans = L_t_batch.transpose(1,2)  # (B,3,3)

                # --- 6. Reparametrize ---
                eps = torch.randn(batch_size, 3, device=device)  # (B,3)
                z_t = mu_t_batch + torch.bmm(eps.unsqueeze(1), L_t_trans).squeeze(1)  # (B,3)

                # --- 7. Target score (full) ---
                rhs = -eps.unsqueeze(-1)  # (B,3,1)
                score_full_target = torch.linalg.solve(L_t_trans, rhs).squeeze(-1)  # (B,3)
                target_ps = score_full_target[:, 1:]  # (B,2)

                # --- 8. Prediction ---
                ps_inputs = z_t[:, 1:]  # (B,2)
                t_idx_tensor = torch.full((batch_size,), t_idx_scalar, device=device, dtype=torch.long)
                score_ps_pred = model(ps_inputs, t_idx_tensor)  # (B,2)

                # --- 9. Reparametrization to eps ---
                score_full_pred = torch.cat([torch.zeros(batch_size,1,device=device), score_ps_pred], dim=1)  # (B,3)
                eps_hat = -torch.bmm(L_t_trans, score_full_pred.unsqueeze(-1)).squeeze(-1)  # (B,3)

                eps_target_ps = eps[:,1:]    # (B,2)
                eps_hat_ps   = eps_hat[:,1:] # (B,2)

                # --- 10. Weighted loss with GG^T ---
                diff = eps_hat_ps - eps_target_ps  # (B,2)
                #per_sample_loss = torch.einsum("bi,ij,bj->b", diff, GGt_ps, diff)  # ||·||_{GG^T}^2
                per_sample_loss = ((eps_hat_ps - eps_target_ps) ** 2).mean(dim=1) # Frobenius

                # Scaling factor: ||L||_{GG^T}^{-2}
                # Norm of L wrt GG^T block
                L_ps = L_t_batch[:,1:,1:]  # (B,2,2)
                #L_norm_sq = torch.einsum("bij,kl,bkl->b", L_ps, GGt_ps, L_ps)  # ||·||_{GG^T}^2
                L_norm_sq = torch.norm(L_ps, dim=(1,2))**2 + 1e-6 # Frobenius

                per_sample_loss = per_sample_loss #/ L_norm_sq
                loss = per_sample_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / n_steps
            losses.append(avg_loss)
            print(f"[GLD HSM] Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")

        return model, losses


