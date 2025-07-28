import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from scipy.signal import convolve
from base import DiffusionModel
from scipy.linalg import expm, eigvals

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Main Implementation ---
class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Implements the forward and reverse processes for a generalized Langevin diffusion SDE.
    The state vector is z = [x, p, s], representing position, momentum, and an auxiliary variable.
    
    Forward SDE: dz = -beta * A * z * dt + G * dW
    where G = sqrt(2 * beta) * B
    """
    def __init__(self, gmm_params, **kwargs):
        super().__init__('Generalized Langevin Diffusion', gmm_params, **kwargs)
        
        # --- Model Parameters ---
        self.beta = 4.0
        self.gamma = 1.0
        self.lambda_val = 0.0 # Using lambda_val to avoid keyword conflict
        self.c = 0.5
        self.M = 0.25
        self.M_inv = 1.0 / self.M

        # Variances for initial momentum and auxiliary variables
        self.p_init_var = 0.01
        self.s_init_var = 0.04

        # --- SDE Matrices (consistent with LaTeX notation) ---
        self.A = torch.tensor([
            [0, -self.M_inv, 0],
            [1, self.M_inv * self.gamma**2, self.gamma * self.lambda_val * self.c],
            [0, self.gamma * self.lambda_val * self.c, self.lambda_val**2]
        ], dtype=torch.float32, device=DEVICE)

        # Base diffusion matrix B (without sqrt(2*beta))
        self.B = torch.tensor([
            [0, 0, 0],
            [0, self.gamma, 0],
            [0, self.gamma * self.lambda_val * self.c, self.lambda_val * np.sqrt(1 - self.c**2)]
        ], dtype=torch.float32, device=DEVICE)
        
        # Full diffusion matrix G = sqrt(2*beta) * B
        self.G = np.sqrt(2 * self.beta) * self.B
        
        # This term is used in the reverse SDE drift
        self.GGt = self.G @ self.G.T
        
        # --- Caches for precomputed values ---
        self.perturbation_cache = {}
        self.precompute() # Automatically precompute on initialization

    def _compute_matrix_exponential_putzer(self, M, t):
        """Computes e^{Mt} using Putzer's algorithm for a single time t."""
        M_np = M.cpu().numpy()
        eigenvalues = eigvals(M_np)
        n = M.shape[0]

        # Ensure eigenvalues are real for simplicity, handle complex case if necessary
        if np.iscomplexobj(eigenvalues):
             # Fallback to scipy's expm if eigenvalues are complex
            return torch.from_numpy(expm(M_np * t)).to(DEVICE)

        eigenvalues = eigenvalues.real

        r = np.zeros(n, dtype=np.float32)
        r[0] = np.exp(eigenvalues[0] * t)
        for i in range(1, n):
            if np.isclose(eigenvalues[i], eigenvalues[i-1]):
                r[i] = t * np.exp(eigenvalues[i] * t)
            else:
                r[i] = (np.exp(eigenvalues[i] * t) - np.exp(eigenvalues[i-1] * t)) / (eigenvalues[i] - eigenvalues[i-1])

        P = [np.eye(n, dtype=np.float32)]
        for i in range(1, n):
            P.append(P[-1] @ (M_np - eigenvalues[i-1] * np.eye(n)))
            
        e_Mt = sum(r[i] * P[i] for i in range(n))
        return torch.from_numpy(e_Mt).to(DEVICE)

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

    def precompute(self):
        """
        Precomputes the evolution of each GMM component's mean and covariance over time
        using the analytical solution from the HOLD paper.
        """
        print("Precomputing perturbation kernels analytically...")
        n_components = len(self.gmm_params['weights'])
        
        F = -self.beta * self.A
        ts_np = self.ts.cpu().numpy()

        for t_idx, t in enumerate(ts_np):
            exp_Ft = self._compute_matrix_exponential_putzer(F, t)
            
            # Covariance integral term
            # For a small dt, we can approximate the integral: integral(0,t) ~= sum(exp(F*s)*GGt*exp(F*s).T*ds)
            # A simpler, more stable approach is to compute it iteratively.
            if t_idx == 0:
                Sigma_integral = torch.zeros_like(self.A)
            else:
                prev_Sigma_integral = self.perturbation_cache[t_idx-1]['Sigma_integral']
                # One step of Euler integration for the integral part
                exp_F_prev_t = self._compute_matrix_exponential_putzer(F, ts_np[t_idx-1])
                integrand = exp_F_prev_t @ self.GGt @ exp_F_prev_t.T
                Sigma_integral = prev_Sigma_integral + integrand * self.dt
            
            means_k = []
            covs_k = []
            for k in range(n_components):
                mu0_k = torch.tensor([self.gmm_params['means'][k], 0, 0], device=DEVICE)
                Sigma0_k = torch.diag(torch.tensor([
                    self.gmm_params['stds'][k]**2, self.p_init_var, self.s_init_var
                ], device=DEVICE))

                # Analytical mean
                mu_t = exp_Ft @ mu0_k
                
                # Analytical covariance
                Sigma_t = exp_Ft @ Sigma0_k @ exp_Ft.T + Sigma_integral
                
                means_k.append(mu_t)
                covs_k.append(Sigma_t)

            self.perturbation_cache[t_idx] = {
                'weights': self.gmm_params['weights'],
                'means': means_k,
                'covs': covs_k,
                'Sigma_integral': Sigma_integral # Cache for next step
            }
        print("Precomputation finished.")


    def _get_perturbed_params(self, t_idx):
        """
        Retrieves the precomputed GMM parameters (weights, means, covs) for a given time step.
        """
        cached_params = self.perturbation_cache[t_idx]
        return cached_params['weights'], cached_params['means'], cached_params['covs']

    def _score_fn(self, z, t_idx):
        """
        Computes the marginal score \nabla_z log p_t(z) for the GMM analytically
        and returns the components for p and s.
        """
        weights, means_k, covs_k = self._get_perturbed_params(t_idx)
        
        batch_size = z.shape[0]
        
        # Tensors to store results for each component
        component_pdfs = torch.zeros(batch_size, len(weights), device=DEVICE)
        component_scores = torch.zeros(batch_size, len(weights), 3, device=DEVICE)

        # Loop through each GMM component
        for k, (mu_k, Sigma_k) in enumerate(zip(means_k, covs_k)):
            # 1. Calculate the PDF of z under the k-th 3D Gaussian component
            try:
                # Add a small epsilon to the diagonal for numerical stability
                stable_Sigma_k = Sigma_k + 1e-6 * torch.eye(3, device=DEVICE)
                dist_3d = torch.distributions.MultivariateNormal(mu_k, stable_Sigma_k)
                log_prob_3d = dist_3d.log_prob(z)
                pdf_k = torch.exp(log_prob_3d)
                
                # 2. Calculate the score of the k-th 3D Gaussian component
                score_k = -torch.linalg.solve(stable_Sigma_k, (z - mu_k).T).T

            except torch.linalg.LinAlgError:
                # Handle cases where covariance might not be positive definite
                pdf_k = torch.zeros(batch_size, device=DEVICE)
                score_k = torch.zeros(batch_size, 3, device=DEVICE)

            component_pdfs[:, k] = pdf_k
            component_scores[:, k, :] = score_k

        # 3. Calculate posterior weights (responsibilities)
        weights_tensor = torch.tensor(weights, device=DEVICE).unsqueeze(0)
        weighted_pdfs = component_pdfs * weights_tensor
        total_pdf = torch.sum(weighted_pdfs, dim=1, keepdim=True)
        responsibilities = weighted_pdfs / (total_pdf + 1e-8) # Shape: [batch_size, n_components]

        # 4. Calculate the final marginal score as the weighted average of component scores
        final_score_3d = torch.sum(responsibilities.unsqueeze(2) * component_scores, dim=1)
        
        # 5. Return only the scores for p and s, as needed by the reverse SDE
        return final_score_3d[:, 1:]


    def solve_reverse_sde(self, zT):
        """
        Solves the reverse SDE dz = [-beta*A*z - G*G^T*S']dt + G*dW_bar using Euler-Maruyama.
        """
        print(f"Solving reverse SDE for {self.name}...")
        
        zs = torch.zeros((zT.shape[0], self.n_steps, 3), device=DEVICE)
        zs[:, -1, :] = zT # Start from time T
        
        sqrt_dt = torch.sqrt(self.dt)

        # Iterate backwards from T to 0
        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]
            
            # 1. Get the conditional score for p and s
            score_ps = self._score_fn(z, i)
            
            # 2. Construct the modified score vector S' = [0, score_p, score_s]
            score_full = torch.zeros_like(z)
            score_full[:, 1:] = score_ps
            
            # 3. Calculate the reverse drift
            drift_f = -self.beta * (self.A @ z.T).T
            score_drift = (self.GGt @ score_full.T).T
            drift_rev = drift_f - score_drift
            
            # 4. Calculate the diffusion term
            dW = torch.randn_like(z) * sqrt_dt
            diffusion_rev = (self.G @ dW.T).T
            
            # 5. Euler-Maruyama step (going backward in time, so z_{t-dt} = z_t - dz)
            dz = drift_rev * self.dt + diffusion_rev
            if i > 0:
                zs[:, i - 1, :] = z - dz
            
        return zs
        
    def run_demonstration(self, n_plot, n_hist):
        """
        Runs and visualizes both the forward and reverse SDE processes.
        """
        print(f"\nRunning demonstration for {self.name}...")
        
        # --- Forward Process ---
        x0 = self._get_initial_samples(n_plot)
        p0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.p_init_var)
        s0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.s_init_var)
        z0 = torch.stack([x0, p0, s0], dim=-1)
        forward_paths = self.solve_forward_sde(z0).cpu().numpy()

        # --- Reverse Process ---
        xT_hist = torch.randn(n_hist, device=DEVICE)
        pT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        sT_hist = torch.randn(n_hist, device=DEVICE)
        zT_hist = torch.stack([xT_hist, pT_hist, sT_hist], dim=1)
        reverse_paths = self.solve_reverse_sde(zT_hist).cpu().numpy()

        # --- Plotting ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        var_names = ['Position (x)', 'Momentum (p)', 'Memory (s)']
        
        ts_cpu = self.ts.cpu().numpy()
        for i in range(3):
            # Plot forward paths
            axes[i, 0].plot(ts_cpu, forward_paths[:, :, i].T, lw=1.5, alpha=0.6)
            axes[i, 0].set_title(f'Forward: {var_names[i]}')
            axes[i, 0].set_ylabel(var_names[i])
            
            # Plot reverse paths
            axes[i, 1].plot(ts_cpu, reverse_paths[:, :, i].T, lw=1.5, alpha=0.6)
            axes[i, 1].set_title(f'Reverse: {var_names[i]}')
            
            if i == 2:
                axes[i, 0].set_xlabel('Time')
                axes[i, 1].set_xlabel('Time')
                axes[i, 2].set_xlabel('Value')

        # Plot final distributions from the reverse process
        plot_position_dist(reverse_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        plot_aux_dist(axes[1, 2], (reverse_paths[:, 0, 1], 'Momentum'), target_dist=(0, np.sqrt(self.p_init_var)))
        plot_aux_dist(axes[2, 2], (reverse_paths[:, 0, 2], 'Memory'), target_dist=(0, np.sqrt(self.s_init_var)))

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()