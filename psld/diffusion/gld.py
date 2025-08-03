import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from scipy.signal import convolve
from base import DiffusionModel
from scipy.linalg import expm, eigvals
from matrix_exp import compute_mean_and_covariance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Main Implementation ---
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
        self.lambda_val = 1.0 # Using lambda_val to avoid keyword conflict
        self.c = 0.5
        self.M = 0.25
        self.M_inv = 1.0 / self.M

        # Variances for initial momentum and auxiliary variables
        self.p_init_var = 0.01
        self.s_init_var = 0.01

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

    def precompute(self):
        """
        Precomputes the evolution of each GMM component's mean and covariance over time
        by calling the accurate analytical solver from matrix_exp.py.
        """
        print("Precomputing perturbation kernels analytically...")
        n_components = len(self.gmm_params['weights'])
        
        # Convert tensors to NumPy arrays once for use with the SciPy-based solver
        A_np = self.A.cpu().numpy()
        G_np = self.G.cpu().numpy()
        ts_np = self.ts.cpu().numpy()

        for t_idx, t in enumerate(ts_np):
            means_k = []
            covs_k = []
            
            for k in range(n_components):
                # Define initial conditions for this GMM component
                mu0_k_np = np.array([self.gmm_params['means'][k], 0, 0])
                Sigma0_k_np = np.diag([
                    self.gmm_params['stds'][k]**2, self.p_init_var, self.s_init_var
                ])

                # Use the imported function for an accurate analytical solution
                mu_t_np, Sigma_t_np = compute_mean_and_covariance(
                    t, self.beta, A_np, G_np, mu0_k_np, Sigma0_k_np
                )
                
                # Convert results back to PyTorch tensors and move to the correct device
                mu_t = torch.from_numpy(mu_t_np).float().to(DEVICE)
                Sigma_t = torch.from_numpy(Sigma_t_np).float().to(DEVICE)
                
                means_k.append(mu_t)
                covs_k.append(Sigma_t)

            # Store the computed values in the cache for this time step
            self.perturbation_cache[t_idx] = {
                'weights': self.gmm_params['weights'],
                'means': means_k,
                'covs': covs_k,
            }
        print("Precomputation finished.")

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

    def solve_forward_analytical(self, z0):
        """
        Simulates the forward process by sampling from the analytical perturbation kernel p(z_t|z_0).
        This serves as a ground truth to verify the correctness of the precomputation.
        """
        print(f"Solving forward process analytically for {self.name}...")
        n_samples = z0.shape[0]
        zs = torch.zeros((n_samples, self.n_steps, 3), device=DEVICE)
        zs[:, 0, :] = z0

        A_np = self.A.cpu().numpy()
        G_np = self.G.cpu().numpy()
        ts_np = self.ts.cpu().numpy()
        
        # The initial covariance for a single point is zero.
        Sigma0_point_np = np.diag([
                sum([self.gmm_params['stds'][k]**2 for k in range(len(self.gmm_params['weights']))]), self.p_init_var, self.s_init_var])

        # Loop over each sample in the batch
        for j in range(n_samples):
            z0_sample_np = z0[j].cpu().numpy()
            for i in range(1, self.n_steps):
                t = ts_np[i]
                
                # Compute the mean and covariance for this specific sample's path
                mu_t_np, Sigma_t_np = compute_mean_and_covariance(
                    t, self.beta, A_np, G_np, z0_sample_np, Sigma0_point_np
                )
                
                # Sample from the resulting multivariate normal distribution
                dist = torch.distributions.MultivariateNormal(
                    torch.from_numpy(mu_t_np).float().to(DEVICE),
                    torch.from_numpy(Sigma_t_np).float().to(DEVICE)
                )
                zs[j, i, :] = dist.sample()
            
        return zs

    def _get_perturbed_params(self, t_idx):
        """
        Retrieves the precomputed GMM parameters (weights, means, covs) for a given time step.
        """
        cached_params = self.perturbation_cache[t_idx]
        return cached_params['weights'], cached_params['means'], cached_params['covs']

    def _score_fn(self, z, t_idx):
        """
        Computes the marginal score \nabla_z log p_t(z) for the GMM analytically
        and returns the components for p and s. This version is general for any lambda_val.
        """
        weights, means_k, covs_k = self._get_perturbed_params(t_idx)
        
        batch_size = z.shape[0]
        
        # Tensors to store results for each component
        log_component_probs = torch.zeros(batch_size, len(weights), device=DEVICE)
        component_scores = torch.zeros(batch_size, len(weights), 3, device=DEVICE)

        # Loop through each GMM component
        for k, (mu_k, Sigma_k) in enumerate(zip(means_k, covs_k)):
            # Add a small epsilon to the diagonal for numerical stability
            stable_Sigma_k = Sigma_k + 1e-6 * torch.eye(3, device=DEVICE)
            dist_3d = torch.distributions.MultivariateNormal(mu_k, stable_Sigma_k)
            log_prob_3d = dist_3d.log_prob(z)
            score_k = -torch.linalg.solve(stable_Sigma_k, (z - mu_k).T).T
            log_component_probs[:, k] = log_prob_3d
            component_scores[:, k, :] = score_k

        # Calculate responsibilities using log-sum-exp for stability
        log_weights = torch.tensor(weights, device=DEVICE).log()
        log_weighted_probs = log_component_probs + log_weights.unsqueeze(0)
        log_total_prob = torch.logsumexp(log_weighted_probs, dim=1, keepdim=True)
        log_responsibilities = log_weighted_probs - log_total_prob
        responsibilities = log_responsibilities.exp()

        # Calculate the final marginal score as the weighted average of component scores
        final_score_3d = torch.sum(responsibilities.unsqueeze(2) * component_scores, dim=1)
        
        # Return only the scores for p and s, as needed by the reverse SDE
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
            
            # 1. Get the score for p and s
            score_ps = self._score_fn(z, i) #torch.zeros_like(z)[:, 1:]
            #score_ps = torch.clamp(score_ps, min=-100.0, max=100.0)
            
            # 2. Construct the modified score vector S' = [0, score_p, score_s]
            score_full = torch.zeros_like(z)
            score_full[:, 1:] = score_ps
            
            # 3. Calculate the reverse drift: f_rev = -f_fwd + G*G^T*S'
            f_fwd = -self.beta * (self.A @ z.T).T
            score_drift = (self.GGt @ score_full.T).T
            drift_rev = -f_fwd + score_drift
            
            # 4. Calculate the diffusion term
            dW = torch.randn_like(z) * sqrt_dt
            diffusion = (self.G @ dW.T).T
            
            # 5. Euler-Maruyama step (going backward in time)
            # z_{t-dt} = z_t - f_rev * dt + G * dW_bar
            if i > 0:
                zs[:, i - 1, :] = z - drift_rev * self.dt + diffusion
            
        return zs
        
    def run_demonstration(self, n_plot, n_hist):
        """
        Runs and visualizes both the forward and reverse SDE processes.
        """
        print(f"\nRunning demonstration for {self.name}...")
        
        # --- Initial Samples ---
        x0 = self._get_initial_samples(n_plot)
        p0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.p_init_var)
        s0 = torch.randn(n_plot, device=DEVICE) * np.sqrt(self.s_init_var)
        z0 = torch.stack([x0, p0, s0], dim=-1)

        # --- Forward Processes ---
        forward_sde_paths = self.solve_forward_sde(z0).cpu().numpy()
        #forward_analytical_paths = self.solve_forward_analytical(z0).cpu().numpy()

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
            #axes[i, 0].plot(ts_cpu, forward_analytical_paths[:, :, i].T, lw=1, alpha=0.8, color='red', linestyle='--', label='Analytical')
            axes[i, 0].plot(ts_cpu, forward_sde_paths[:, :, i].T, lw=1.5, alpha=0.6, color='blue', label='SDE Sim')
            #axes[i, 0].legend()

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