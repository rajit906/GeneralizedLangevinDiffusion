import torch
import numpy as np
import scipy.linalg
from tqdm import tqdm

from .base import DiffusionModel
from matrix_exp import stationary_covariance, compute_mean_and_covariance
import torch.optim as optim

class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Implements the forward and reverse processes for a generalized Langevin 
    diffusion SDE, generalized to arbitrary data dimension `d`.
    
    The state vector is z = [x, p, s], where each is in R^d.
    Total state dimension is 3*d.
    """
    def __init__(self, data_dim, inttype='em', gamma=1.0, lambda_val=1.0, 
                 c_val=0.1, M=1., device=torch.device("cpu"), **kwargs):
        
        super().__init__('Generalized Langevin Diffusion', data_dim, **kwargs)
        self.device = device
        
        # --- Model Parameters ---
        self.inttype = inttype
        self.gamma = gamma
        self.c = c_val
        self.lambda_val = lambda_val
        self.M = M
        self.M_inv = 1. / self.M
        self.beta = 8. * np.sqrt(self.M) # SDE parameter
        self.dim = 3 * self.data_dim     # Total state dimension

        # Initial variances for p_0 and s_0
        self.p_init_var = 1.0
        self.s_init_var = 1.0

        # --- Build Generalized Block Matrices (3d x 3d) ---
        d = self.data_dim
        I_d = torch.eye(d, device=self.device)
        O_d = torch.zeros((d, d), device=self.device)

        # Hamiltonian (skew-symmetric) part
        self.AH = torch.zeros((self.dim, self.dim), device=self.device)
        self.AH[:d, d:2*d] = -self.M_inv * I_d # -M_inv
        self.AH[d:2*d, :d] = I_d              # +I

        # Damping (symmetric) part
        self.AGamma = torch.zeros((self.dim, self.dim), device=self.device)
        self.AGamma[d:2*d, d:2*d] = (self.M_inv * self.gamma**2) * I_d
        self.AGamma[d:2*d, 2*d:] = (self.gamma * self.lambda_val * self.c) * I_d
        self.AGamma[2*d:, d:2*d] = (self.gamma * self.lambda_val * self.c) * I_d
        self.AGamma[2*d:, 2*d:] = (self.lambda_val**2) * I_d
        
        # Full Drift Matrix
        self.A = (self.AH + self.AGamma).to(self.device)

        # Noise Covariance Matrix
        self.B = torch.zeros((self.dim, self.dim), device=self.device)
        self.B[d:2*d, d:2*d] = self.gamma * I_d
        self.B[d:2*d, 2*d:] = (self.lambda_val * self.c) * I_d
        self.B[2*d:, 2*d:] = (self.lambda_val * np.sqrt(1 - self.c**2)) * I_d
        self.B = self.B.to(self.device)

        self.G = np.sqrt(2 * self.beta) * self.B
        self.GGt = (self.G @ self.G.T).to(self.device)

        # Numpy versions for scipy
        self.A_np = self.A.cpu().numpy()
        self.G_np = self.G.cpu().numpy()
        
        # Stationary covariance (solution to Lyapunov equation)
        self.C_np = stationary_covariance(self.beta, self.A_np, self.G_np)
        
        # Placeholders for precomputed kernels
        self.M_t_all = []
        self.L_t_all = []


    def precompute(self):
        """
        Precomputes the transition kernel p(z_t | z_0) = N(M_t z_0, Sigma_t)
        for all t in self.ts.
        
        mu_t(z_0) = M_t @ z_0, where M_t = expm(-beta * A * t)
        Sigma_t = C - M_t @ C @ M_t.T, where C is stationary cov.
        We store M_t and L_t (Cholesky of Sigma_t).
        """
        print("Precomputing transition kernels...")
        C_np = self.C_np
        F_base = -self.beta * self.A_np # Full drift F = -beta * A
        
        for t in tqdm(self.ts.cpu().numpy(), desc="Precomputing"):
            M_t_np = scipy.linalg.expm(F_base * t)
            Sigma_t_np = C_np - M_t_np @ C_np @ M_t_np.T
            
            # Add jitter for numerical stability
            jitter = 1e-9 * np.eye(self.dim)
            L_t_np = np.linalg.cholesky(Sigma_t_np + jitter)
            
            self.M_t_all.append(torch.from_numpy(M_t_np).float().to(self.device))
            self.L_t_all.append(torch.from_numpy(L_t_np).float().to(self.device))
            
        print("...Precomputation complete.")


    def sample_prior(self, n_samples):
        """Samples z_T from the prior (stationary) distribution N(0, C)."""
        # Use Cholesky of stationary covariance
        try:
            L_T_np = np.linalg.cholesky(self.C_np + 1e-9 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            print("Warning: Prior covariance not positive definite. Using sqrtm.")
            # Fallback to matrix square root (slower)
            L_T_np = scipy.linalg.sqrtm(self.C_np + 1e-9 * np.eye(self.dim))
            
        L_T = torch.from_numpy(L_T_np).float().to(self.device)
        
        zT = (L_T @ torch.randn(n_samples, self.dim, device=self.device).T).T
        return zT

    
    def solve_reverse_sde_em(self, zT, score_model):
        """
        Solves the reverse-time SDE:
            dz = [f(z,t) - G G^T s(z,t)] dt + G dW_bar,
        integrated backward with Euler–Maruyama.
        
        NOTE: This now REQUIRES a score_model.
        """
        B = zT.shape[0]
        zs = torch.zeros((B, self.n_steps, self.dim), device=self.device)
        zs[:, -1, :] = zT
        sqrt_dt = torch.sqrt(self.dt)
        d = self.data_dim

        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]
            t_idx = i
            t_val = self.ts[t_idx].expand(B).to(self.device)

            # --- Compute score from network ---
            # Model only predicts score for (p, s)
            ps_inputs = z[:, d:] # (B, 2d)
            score_ps = score_model(ps_inputs, t_val) # (B, 2d)
            
            # score_full = [0_x, score_p, score_s]
            score_full = torch.cat(
                [torch.zeros(B, d, device=self.device), score_ps], 
                dim=1
            ) # (B, 3d)

            # --- Forward drift ---
            f_fwd = -self.beta * (self.A @ z.T).T

            # --- Reverse drift ---
            score_drift = (self.GGt @ score_full.T).T
            drift_rev = f_fwd - score_drift

            # --- Noise term ---
            dW = torch.randn_like(z) * sqrt_dt
            diffusion = (self.G @ dW.T).T

            # --- Backward integration step ---
            if i > 0:
                zs[:, i - 1, :] = z - drift_rev * self.dt + diffusion

        return zs
   
    
    def solve_pfode(self, zT, score_model):
        """
        Solve reverse-time Probability Flow ODE:
            dz = [ f(z,t) - 0.5 * G G^T * score(z,t) ] dt
        
        NOTE: This now REQUIRES a score_model.
        """
        B = zT.shape[0]
        zs = torch.zeros((B, self.n_steps, self.dim), device=self.device)
        zs[:, -1, :] = zT
        d = self.data_dim

        ts = self.ts.to(self.device)
        dt_back = torch.empty(self.n_steps, device=self.device, dtype=ts.dtype)
        dt_back[0] = ts[0]  # Not used
        dt_back[1:] = ts[1:] - ts[:-1]

        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :]  # (B, 3d)
            t_val = ts[i].expand(B)

            # --- Compute score from network ---
            ps_inputs = z[:, d:]  # (B, 2d)
            score_ps = score_model(ps_inputs, t_val)  # (B, 2d)
            score_full = torch.cat(
                [torch.zeros(B, d, device=self.device), score_ps], 
                dim=1
            ) # (B, 3d)

            # --- Forward drift ---
            f_fwd = -self.beta * (self.A @ z.T).T  # (B, 3d)

            # --- Deterministic ODE drift ---
            score_drift = (self.GGt @ score_full.T).T  # (B, 3d)
            drift_ode = f_fwd - 0.5 * score_drift

            # --- Integrate backward ---
            if i > 0:
                zs[:, i - 1, :] = z - drift_ode * dt_back[i]

        return zs


    def generate_samples(self, n_samples, score_model, method='pfode'):
        """
        Generates samples by solving the reverse process.
        
        Args:
            n_samples (int): Number of samples to generate.
            score_model (nn.Module): The trained score network.
            method (str): 'pfode' (default) or 'em' (SDE).
        
        Returns:
            torch.Tensor: Generated paths of shape (n_samples, n_steps, 3*d)
        """
        zT = self.sample_prior(n_samples)
        
        if method == 'pfode':
            paths = self.solve_pfode(zT, score_model)
        elif method == 'em':
            paths = self.solve_reverse_sde_em(zT, score_model)
        else:
            raise ValueError(f"Unknown generation method: {method}")
            
        return paths

    
    def train_score_network(self, ScoreNetwork, dataloader, n_epochs=50, lr=1e-3, val_dataloader=None):
        """
        Trains the score network using Denoising Score Matching on the
        precomputed transition kernel p(z_t | z_0).
        """
        d = self.data_dim
        model = ScoreNetwork(data_dim=d).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Get the (p,s) block of GGt for the loss
        GGt_ps = self.GGt[d:, d:] # (2d, 2d)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            
            # --- Training Loop ---
            model.train()
            total_train_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
            
            for x0_batch in pbar:
                B = x0_batch.shape[0]
                if B == 0: continue
                
                # 1. Sample z0 = [x0, p0, s0]
                x0 = x0_batch.to(self.device).view(B, d)
                p0 = torch.randn(B, d, device=self.device) * np.sqrt(self.p_init_var)
                s0 = torch.randn(B, d, device=self.device) * np.sqrt(self.s_init_var)
                z0 = torch.cat([x0, p0, s0], dim=1) # (B, 3d)

                # 2. Sample single time index t
                t_idx = int(torch.randint(0, self.n_steps, (1,)).item())
                t_scalar = self.ts[t_idx]
                t_batch = t_scalar.expand(B).to(self.device)

                # 3. Get precomputed kernels M_t, L_t
                M_t = self.M_t_all[t_idx]
                L_t = self.L_t_all[t_idx]
                L_t_trans = L_t.T

                # 4. Perturb: z_t = M_t z_0 + L_t eps
                eps = torch.randn(B, self.dim, device=self.device)  # (B, 3d)
                mu_t = (M_t @ z0.T).T
                z_t = mu_t + (L_t @ eps.T).T

                # 5. Target score (reparametrized to eps)
                # target_score = - (Sigma_t)^-1 * (z_t - mu_t)
                #              = - (L_t L_t^T)^-1 * L_t eps
                #              = - (L_t^T)^-1 * eps
                # We target eps directly.
                eps_target_ps = eps[:, d:]    # (B, 2d)

                # 6. Prediction
                ps_inputs = z_t[:, d:]  # (B, 2d)
                score_ps_pred = model(ps_inputs, t_batch) # (B, 2d)

                # 7. Reparametrize prediction to eps_hat
                # score_full_pred = [0, score_ps_pred]
                score_full_pred = torch.cat(
                    [torch.zeros(B, d, device=self.device), score_ps_pred], 
                    dim=1
                ) # (B, 3d)
                
                # eps_hat = - L_t^T @ score_pred
                eps_hat = -torch.bmm(
                    L_t_trans.expand(B, -1, -1), 
                    score_full_pred.unsqueeze(-1)
                ).squeeze(-1) # (B, 3d)
                
                eps_hat_ps = eps_hat[:, d:] # (B, 2d)

                # 8. Weighted loss with GGt
                diff = eps_hat_ps - eps_target_ps  # (B, 2d)
                
                # || diff ||^2_{GGt_ps}
                per_sample_loss = torch.einsum("bi,ij,bj->b", diff, GGt_ps, diff)
                
                # 9. Loss scaling factor
                L_ps = L_t[d:, d:]  # (2d, 2d)
                # ||L_ps||^2_{GGt_ps}
                L_norm_sq_scalar = torch.einsum("ij,kl,kl->", L_ps, GGt_ps, L_ps)
                
                loss = (per_sample_loss / (L_norm_sq_scalar + 1e-8)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            avg_train_loss = total_train_loss / len(dataloader)
            train_losses.append(avg_train_loss)

            # --- Validation Loop ---
            if val_dataloader:
                model.eval()
                total_val_loss = 0.0
                pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]")
                with torch.no_grad():
                    for x0_batch in pbar_val:
                        B = x0_batch.shape[0]
                        if B == 0: continue
                        
                        # (Repeat steps 1-9 without backprop)
                        x0 = x0_batch.to(self.device).view(B, d)
                        p0 = torch.randn(B, d, device=self.device) * np.sqrt(self.p_init_var)
                        s0 = torch.randn(B, d, device=self.device) * np.sqrt(self.s_init_var)
                        z0 = torch.cat([x0, p0, s0], dim=1) 
                        
                        t_idx = int(torch.randint(0, self.n_steps, (1,)).item())
                        t_scalar = self.ts[t_idx]
                        t_batch = t_scalar.expand(B).to(self.device)
                        
                        M_t = self.M_t_all[t_idx]
                        L_t = self.L_t_all[t_idx]
                        L_t_trans = L_t.T
                        
                        eps = torch.randn(B, self.dim, device=self.device)
                        mu_t = (M_t @ z0.T).T
                        z_t = mu_t + (L_t @ eps.T).T
                        
                        eps_target_ps = eps[:, d:]
                        
                        ps_inputs = z_t[:, d:]
                        score_ps_pred = model(ps_inputs, t_batch)
                        
                        score_full_pred = torch.cat(
                            [torch.zeros(B, d, device=self.device), score_ps_pred], 
                            dim=1
                        )
                        eps_hat = -torch.bmm(
                            L_t_trans.expand(B, -1, -1), 
                            score_full_pred.unsqueeze(-1)
                        ).squeeze(-1)
                        eps_hat_ps = eps_hat[:, d:]
                        
                        diff = eps_hat_ps - eps_target_ps
                        per_sample_loss = torch.einsum("bi,ij,bj->b", diff, GGt_ps, diff)
                        L_ps = L_t[d:, d:]
                        L_norm_sq_scalar = torch.einsum("ij,kl,kl->", L_ps, GGt_ps, L_ps)
                        
                        loss = (per_sample_loss / (L_norm_sq_scalar + 1e-8)).mean()
                        total_val_loss += loss.item()
                        pbar_val.set_postfix(loss=loss.item())

                avg_val_loss = total_val_loss / len(val_dataloader)
                val_losses.append(avg_val_loss)
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

        return model, (train_losses, val_losses)