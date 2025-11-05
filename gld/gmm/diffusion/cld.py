# TODO:
# Vectorize. Remove MVN. Remove inverse.
# cld.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from viz import plot_aux_dist, plot_position_dist
from base import DiffusionModel
from scipy.integrate import solve_ivp
import scipy.linalg
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cpu")

class CriticallyDampedLangevin(DiffusionModel):
    """Implements Critically Damped Langevin Diffusion (CLD)."""
    def __init__(self, gmm_params, **kwargs):
        self.M = 0.25
        self.Gamma = 1.
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
        n_batch, dim = z.shape
        n_comp = len(weights)

        # Stack mixture parameters
        means = torch.stack(means, dim=0)        # (n_comp, dim)
        covs = torch.stack(covs, dim=0)          # (n_comp, dim, dim)
        covs = covs + 1e-6 * torch.eye(dim, device=DEVICE)[None]  # stabilization
        weights = torch.as_tensor(weights, device=DEVICE)         # (n_comp,)

        # Expand z to (n_batch, n_comp, dim)
        z_exp = z[:, None, :]                     # (n_batch, 1, dim)
        diff = z_exp - means[None, :, :]          # (n_batch, n_comp, dim)

        # Precompute inverse and logdet of covariances
        cov_inv = torch.linalg.inv(covs)          # (n_comp, dim, dim)
        cov_logdet = torch.logdet(covs)           # (n_comp,)

        # Mahalanobis distances: (n_batch, n_comp)
        m_dist = torch.einsum("bkd,kde,bke->bk", diff, cov_inv, diff)
        
        # log pdfs: (n_batch, n_comp)
        log_pdf = -0.5 * (m_dist + dim*np.log(2*np.pi) + cov_logdet[None, :])
        pdf = torch.exp(log_pdf)                  # (n_batch, n_comp)

        # Gradient of log pdf wrt z: - (cov_inv @ diff.T).T
        grad_log_pdf = -torch.einsum("kde,bke->bkd", cov_inv, diff)  # (n_batch, n_comp, dim)

        # Weighted mixture contributions
        w_pdf = weights[None, :] * pdf            # (n_batch, n_comp)
        p_t_z = w_pdf.sum(dim=1)                  # (n_batch,)
        grad_v_p_t_z = (w_pdf[..., None] * grad_log_pdf)[..., 1].sum(dim=1)  # (n_batch,)

        # Final score (zeros except v-component)
        score_full = torch.zeros_like(z)          # (n_batch, dim)
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

    def solve_reverse_sde_em(self, zT, score_model=None):
        """
        Reverse SDE (Euler–Maruyama), optionally using learned score (velocity only).
        """
        zs = torch.zeros(zT.shape[0], self.n_steps, 2, device=DEVICE)
        zs[:, -1, :] = zT
        sqrt_dt = torch.sqrt(self.dt)

        with torch.no_grad():
            for i in range(self.n_steps - 1, -1, -1):
                z = zs[:, i, :]

                if score_model is None:
                    score_full = self._score_fn(z, i)
                    score_v = score_full[:, 1]
                else:
                    v = z[:, 1:2]  # (B,1)
                    t_idx = torch.full((z.shape[0],), i, device=DEVICE, dtype=torch.long)
                    score_v = score_model(v, t_idx).squeeze(-1)

                f_fwd = -self.beta * (self.A @ z.T).T
                score_drift = (self.GGt @ torch.stack([torch.zeros_like(score_v), score_v], dim=1).T).T
                drift_rev = -f_fwd + score_drift
                dW = torch.randn_like(z) * sqrt_dt
                diffusion = (self.G @ dW.T).T
                if i > 0:
                    zs[:, i-1, :] = z - drift_rev * self.dt + diffusion

        return zs


    
    def solve_reverse_sde_sscs(self, zT, score_model=None):
        """
        Reverse SDE using SSCS scheme, optionally with learned score (velocity only).
        """
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

        with torch.no_grad():
            for i in range(self.n_steps - 1, 0, -1):
                u_n = zs[:, i, :]

                # half-step mean
                mu_x_half = (2*B_half_dt/self.Gamma*u_n[:,0] - 4*B_half_dt/self.Gamma**2*u_n[:,1] + u_n[:,0]) * exp_term
                mu_v_half = (B_half_dt*u_n[:,0] - 2*B_half_dt/self.Gamma*u_n[:,1] + u_n[:,1]) * exp_term
                mu_half = torch.stack([mu_x_half, mu_v_half], dim=1)

                u_half = torch.distributions.MultivariateNormal(mu_half, cov_half).sample()

                # --- Score (velocity only) ---
                if score_model is None:
                    score_full = self._score_fn(u_half, i)
                    score_v = score_full[:, 1]
                else:
                    v = u_half[:, 1:2]
                    t_idx = torch.full((u_half.shape[0],), i, device=DEVICE, dtype=torch.long)
                    score_v = score_model(v, t_idx).squeeze(-1)

                # velocity update
                v_update = self.dt * (2 * self.beta * self.Gamma * (score_v + M_inv * u_half[:,1]))
                u_half_prime = u_half.clone()
                u_half_prime[:, 1] += v_update

                # full-step mean
                mu_x_full = (2*B_half_dt/self.Gamma*u_half_prime[:,0] - 4*B_half_dt/self.Gamma**2*u_half_prime[:,1] + u_half_prime[:,0]) * exp_term
                mu_v_full = (B_half_dt*u_half_prime[:,0] - 2*B_half_dt/self.Gamma*u_half_prime[:,1] + u_half_prime[:,1]) * exp_term
                mu_full = torch.stack([mu_x_full, mu_v_full], dim=1)

                zs[:, i-1, :] = torch.distributions.MultivariateNormal(mu_full, cov_half).sample()

        return zs
    
    def solve_reverse_sde_ubu(self, zT):
        return None
    
    def solve_reverse_sde(self, zT, type='sscs', score_model=None):
        if type=='em':
            return self.solve_reverse_sde_em(zT, score_model=score_model)
        elif type == 'sscs':
            return self.solve_reverse_sde_sscs(zT, score_model=score_model)
        elif type == 'ubu':
            return self.solve_reverse_sde_ubu(zT, score_model=score_model)

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


    def run_demonstration(self, n_plot, n_hist, score_model=None):
        x0_plot = self._get_initial_samples(n_plot)
        z0_plot = torch.stack([x0_plot, torch.randn(n_plot, device=DEVICE) * np.sqrt(self.v_init_var)], dim=1)
        xT_hist = torch.randn(n_hist, device=DEVICE)
        vT_hist = torch.randn(n_hist, device=DEVICE) * np.sqrt(self.M)
        zT_hist = torch.stack([xT_hist, vT_hist], dim=1)
        
        forward_paths = self.solve_forward_sde(z0_plot).cpu().numpy()
        reverse_sde_paths = self.solve_reverse_sde(zT_hist, type='sscs', score_model=score_model).cpu().numpy()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.name} Demonstration', fontsize=16)
        
        ts_cpu = self.ts.cpu()
        
        # --- Position Plots ---
        # Forward trajectories
        axes[0, 0].plot(ts_cpu, forward_paths[10:, :, 0].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[0, 0].plot(ts_cpu, forward_paths[:10, :, 0].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[0, 0].set_title('Forward: Position')
        axes[0, 0].set_ylabel('Position')
        #axes[0, 0].set_ylim(-60, 60)

        # Reverse trajectories
        axes[0, 1].plot(ts_cpu, reverse_sde_paths[10:n_plot, :, 0].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[0, 1].plot(ts_cpu, reverse_sde_paths[:10, :, 0].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[0, 1].set_title('Reverse: Position')
        #axes[0, 1].set_ylim(-60, 60)

        # Final distribution
        plot_position_dist(reverse_sde_paths[:, 0, 0], self.gmm_params, axes[0, 2])
        axes[0, 2].set_title("Final Position Distribution")
        #axes[0, 2].set_xlim(-60, 60)

        # --- Momentum Plots ---
        # Forward trajectories
        axes[1, 0].plot(ts_cpu, forward_paths[10:, :, 1].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[1, 0].plot(ts_cpu, forward_paths[:10, :, 1].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[1, 0].set_title('Forward: Momentum')
        axes[1, 0].set_xlabel('Time')
        axes[1, 0].set_ylabel('Momentum')
        #axes[1, 0].set_ylim(-60, 60)

        # Reverse trajectories
        axes[1, 1].plot(ts_cpu, reverse_sde_paths[10:n_plot, :, 1].T, lw=1.5, color='darkblue', alpha=0.05)
        axes[1, 1].plot(ts_cpu, reverse_sde_paths[:10, :, 1].T, lw=1.5, color='darkblue', alpha=1.0)
        axes[1, 1].set_title('Reverse: Momentum')
        axes[1, 1].set_xlabel('Time')
        #axes[1, 1].set_ylim(-60, 60)
        
        # Final distribution
        plot_aux_dist(axes[1, 2], (reverse_sde_paths[:, 0, 1], 'Momentum'), target_dist=(0, np.sqrt(self.v_init_var)))
        axes[1, 2].set_title("Final Momentum Distribution")
        #axes[1, 2].set_xlim(-40, 40)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def train_score_network_hsm(self, ScoreNetwork, n_epochs=50, batch_size=128, lr=1e-3, n_steps=1000, eps_stabilize=1e-6):
        """
        Hybrid Score Matching (HSM) training implemented correctly.

        - Uses closed-form µ_t(x0) = M_t @ [x0, 0] and covariance Σ_t for p_t(u_t | x0)
        (appendix B.1/B.3 formulas).
        - Reparameterizes the regression loss as noise-prediction but the model still
        predicts the velocity-score s_theta (not alpha). Concretely:
            target_score_v = -`_t * eps_v
        and we minimize MSE(s_pred_v, target_score_v).
        - By default uses the 'drop-variance' weighting λ(t) = `_t^-2 (paper suggestion).
        Set lambda_mode='ml' to use ML weighting (Γβ) or 'plain' to use λ=1.
        """
        device = DEVICE
        model = ScoreNetwork().to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction='mean')

        losses = []

        for epoch in range(n_epochs):
            total_loss = 0.0

            for _ in range(n_steps):
                # 1) sample batch of x0 (data) and a single t uniformly (shared across batch)
                x0 = self._get_initial_samples(batch_size).to(device)  # shape (B,)
                t_idx = int(torch.randint(0, self.n_steps, (1,)).item())  # single t per step
                t = float(self.ts[t_idx].item())                          # scalar time
                B_t = self.beta * t

                # 2) mean propagator M_t (2x2) and batch means µ_t(x0)
                M_t = self.M_ts[t_idx]               # (2,2), torch
                mu0 = torch.stack([x0, torch.zeros_like(x0)], dim=1)  # (B,2)
                # mu_t for each sample: (B,2) = mu0 @ M_t.T
                mu_t = mu0 @ M_t.T  # shape (B,2)

                # 3) analytic covariance Σ_t for conditioning on x0 (Sigma_xx0 = 0, Sigma_vv0 = gamma_init * M)
                #    formulas follow App. B.1 (same algebra as your _get_perturbed_params but with Sigma0_xx = 0).
                Sigma0_xx = 0.0
                Sigma0_vv = float(self.gamma_init * self.M)
                # compute scalar quantities (use python floats for stability; convert to torch later)
                exp_term_cov = np.exp(4.0 * B_t / self.Gamma)

                Sigma_t_xx = (Sigma0_xx
                            + exp_term_cov - 1.0
                            + (4.0 * B_t / self.Gamma) * (Sigma0_xx - 1.0)
                            + (4.0 * B_t**2 / self.Gamma**2) * (Sigma0_xx - 2.0)
                            + (16.0 * B_t**2 / self.Gamma**4) * Sigma0_vv)

                Sigma_t_xv = (-B_t * Sigma0_xx
                            + (4.0 * B_t / self.Gamma**2) * Sigma0_vv
                            - (2.0 * B_t**2 / self.Gamma) * (Sigma0_xx - 2.0)
                            - (8.0 * B_t**2 / self.Gamma**3) * Sigma0_vv)

                Sigma_t_vv = ((self.Gamma**2 / 4.0) * (exp_term_cov - 1.0)
                            + B_t * self.Gamma
                            + Sigma0_vv * (1.0 + 4.0 * B_t**2 / self.Gamma**2 - 4.0 * B_t / self.Gamma)
                            + B_t**2 * (Sigma0_xx - 2.0))

                cov_hat = np.array([[Sigma_t_xx, Sigma_t_xv],
                                    [Sigma_t_xv, Sigma_t_vv]], dtype=np.float32)

                cov_t_np = np.exp(-4.0 * B_t / self.Gamma) * cov_hat   # outer exponential factor
                cov_t = torch.from_numpy(cov_t_np).to(device)          # (2,2) torch

                # numeric stabilization and cholesky
                cov_t = cov_t + eps_stabilize * torch.eye(2, device=device)
                try:
                    L_t = torch.linalg.cholesky(cov_t)  # (2,2)
                except RuntimeError:
                    # fallback to slightly larger stabilization if cholesky fails
                    cov_t = cov_t + (1e-6 * torch.eye(2, device=device))
                    L_t = torch.linalg.cholesky(cov_t)

                # 4) sample epsilon and construct u_t via reparameterization: u_t = µ_t + L_t @ eps
                eps_noise = torch.randn(batch_size, 2, device=device)  # (B,2), standard normal
                u_t = mu_t + eps_noise @ L_t.T                         # (B,2)

                # 5) compute `t (ell_t) used in HSM reparam: `t = sqrt( Σ_xx / (Σ_xx Σ_vv - Σ_xv^2) )
                Σ_xx = cov_t[0, 0]
                Σ_xv = cov_t[0, 1]
                Σ_vv = cov_t[1, 1]
                denom = (Σ_xx * Σ_vv - Σ_xv * Σ_xv).clamp(min=1e-12)
                ell_t = torch.sqrt((Σ_xx / denom).clamp(min=1e-12))  # scalar tensor

                # 6) target velocity-score: ∇_v log p_t(u_t | x0) = -`_t * eps_v
                #target_v = -ell_t * eps_noise[:, 1]   # shape (B,)

                # 7) model prediction: model expects (v, t_idx) like elsewhere in your code
                v_input = u_t[:, 1:2]  # shape (B,1)
                t_idx_batch = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
                scores_pred_v = model(v_input, t_idx_batch).squeeze(-1)  # (B,)

                # --- Noise prediction reparam (Appendix B.3) ---
                eps_v = eps_noise[:, 1]              # true Gaussian noise

                # Convert to equivalent noise prediction
                pred_noise = -(1.0 / (ell_t + 1e-12)) * scores_pred_v  # (B,)

                # 8) weighting λ(t): follow paper's "drop variance" by default (λ = `_t^-2)
                lambda_t = (1.0 / (ell_t**2 + 1e-12)).to(device)  # scalar tensor

                # 9) loss (MSE between predicted velocity-score and analytic target)
                loss = lambda_t * loss_fn(pred_noise, eps_v) #loss_fn(scores_pred_v, target_v)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())

            avg_loss = total_loss / n_steps
            losses.append(avg_loss)
            print(f"[HSM] Epoch {epoch+1}/{n_epochs} - Loss: {avg_loss:.6f}")

        return model, losses


    