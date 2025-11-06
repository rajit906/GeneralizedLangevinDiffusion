import torch
import numpy as np
import scipy.linalg
from tqdm import tqdm

from .base import DiffusionModel
from matrix_exp import stationary_covariance, compute_mean_and_covariance
import torch.optim as optim

class GeneralizedLangevinDiffusion(DiffusionModel):
    """
    Generalized to ND data (images) by applying the SDE per-pixel.
    'data_dim' is now interpreted as 'channel_dim' (C).
    The state vector is z = [x, p, s], where each is (B, C, H, W).
    Total state dimension is 3*C channels.
    """
    def __init__(self, data_dim, inttype='em', gamma=1.0, lambda_val=1.0, 
                 c_val=0.1, M=1., device=torch.device("cpu"), **kwargs):
        
        # For images, data_dim is channel_dim (e.g., 1 for MNIST)
        super().__init__('Generalized Langevin Diffusion', data_dim, **kwargs)
        self.device = device
        self.channel_dim = data_dim # Renaming for clarity
        
        # --- Model Parameters ---
        self.inttype = inttype
        self.gamma = gamma
        self.c = c_val
        self.lambda_val = lambda_val
        self.M = M
        self.M_inv = 1. / self.M
        self.beta = 8. * np.sqrt(self.M)
        self.dim = 3 * self.channel_dim # Total channels: [x, p, s]
        
        self.p_init_var = 1.0
        self.s_init_var = 1.0

        # --- Build SMALL (3C x 3C) Block Matrices ---
        # For MNIST, C=1, so these are (3, 3)
        C = self.channel_dim
        I_c = torch.eye(C, device=self.device)
        O_c = torch.zeros((C, C), device=self.device)

        self.AH = torch.zeros((self.dim, self.dim), device=self.device)
        self.AH[:C, C:2*C] = -self.M_inv * I_c
        self.AH[C:2*C, :C] = I_c

        self.AGamma = torch.zeros((self.dim, self.dim), device=self.device)
        self.AGamma[C:2*C, C:2*C] = (self.M_inv * self.gamma**2) * I_c
        self.AGamma[C:2*C, 2*C:] = (self.gamma * self.lambda_val * self.c) * I_c
        self.AGamma[2*C:, C:2*C] = (self.gamma * self.lambda_val * self.c) * I_c
        self.AGamma[2*C:, 2*C:] = (self.lambda_val**2) * I_c
        
        self.A = (self.AH + self.AGamma).to(self.device)

        self.B = torch.zeros((self.dim, self.dim), device=self.device)
        self.B[C:2*C, C:2*C] = self.gamma * I_c
        self.B[C:2*C, 2*C:] = (self.lambda_val * self.c) * I_c
        self.B[2*C:, 2*C:] = (self.lambda_val * np.sqrt(1 - self.c**2)) * I_c
        self.B = self.B.to(self.device)

        self.G = np.sqrt(2 * self.beta) * self.B
        self.GGt = (self.G @ self.G.T).to(self.device)

        self.A_np = self.A.cpu().numpy()
        self.G_np = self.G.cpu().numpy()
        
        # This is now a small (3C, 3C) matrix, computation is fast
        self.C_np = stationary_covariance(self.beta, self.A_np, self.G_np)
        
        self.M_t_all = []
        self.L_t_all = []


    def precompute(self):
        """
        Precomputes the transition kernel p(z_t | z_0) = N(M_t z_0, Sigma_t)
        for all t in self.ts.
        Since we use a per-pixel SDE, all matrices are small (3C, 3C).
        This is very fast.
        """
        print("Precomputing transition kernels (small 3C x 3C matrices)...")
        C_np = self.C_np
        F_base = -self.beta * self.A_np
        
        for t in tqdm(self.ts.cpu().numpy(), desc="Precomputing"):
            M_t_np = scipy.linalg.expm(F_base * t)
            Sigma_t_np = C_np - M_t_np @ C_np @ M_t_np.T
            jitter = 1e-9 * np.eye(self.dim)
            L_t_np = np.linalg.cholesky(Sigma_t_np + jitter)
            
            self.M_t_all.append(torch.from_numpy(M_t_np).float().to(self.device))
            self.L_t_all.append(torch.from_numpy(L_t_np).float().to(self.device))
            
        print("...Precomputation complete.")


    def sample_prior(self, n_samples, img_shape):
        """
        Samples z_T from the prior (stationary) distribution N(0, C)
        for a given image shape.
        img_shape: (C, H, W)
        """
        C, H, W = img_shape
        try:
            L_T_np = np.linalg.cholesky(self.C_np + 1e-9 * np.eye(self.dim))
        except np.linalg.LinAlgError:
            L_T_np = scipy.linalg.sqrtm(self.C_np + 1e-9 * np.eye(self.dim))
            
        L_T = torch.from_numpy(L_T_np).float().to(self.device) # (3C, 3C)
        
        # Sample standard normal noise in image shape
        eps = torch.randn(n_samples, self.dim, H, W, device=self.device) # (B, 3C, H, W)
        
        # Apply L_T across channel dim
        zT = torch.einsum('ij,bcjhw->bcihw', L_T, eps)
        return zT

    
    def solve_reverse_sde_em(self, zT, score_model):
        """ Solves the reverse-time SDE for image data. """
        B, _, H, W = zT.shape
        zs = torch.zeros((B, self.n_steps, self.dim, H, W), device=self.device)
        zs[:, -1, :, :, :] = zT
        sqrt_dt = torch.sqrt(self.dt)
        C = self.channel_dim

        # Pre-calculate for speed
        A_T = self.A.T
        G_T = self.G.T

        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :, :, :] # (B, 3C, H, W)
            t_val = self.ts[i].expand(B).to(self.device)

            # --- Compute score from network ---
            ps_inputs = z[:, C:, :, :] # (B, 2C, H, W)
            score_ps = score_model(ps_inputs, t_val) # (B, 2C, H, W)
            
            score_full = torch.cat(
                [torch.zeros(B, C, H, W, device=self.device), score_ps], 
                dim=1
            ) # (B, 3C, H, W)

            # --- Forward drift: f_fwd = -beta * (A @ z.T).T ---
            # We use einsum to multiply (3C, 3C) A with (B, 3C, H, W) z
            f_fwd = -self.beta * torch.einsum('ij,bcjhw->bcihw', self.A, z)

            # --- Reverse drift: score_drift = (GGt @ score_full.T).T ---
            score_drift = torch.einsum('ij,bcjhw->bcihw', self.GGt, score_full)
            drift_rev = f_fwd - score_drift

            # --- Noise term: diffusion = (G @ dW.T).T ---
            dW = torch.randn_like(z) * sqrt_dt # (B, 3C, H, W)
            diffusion = torch.einsum('ij,bcjhw->bcihw', self.G, dW)

            if i > 0:
                zs[:, i - 1, :, :, :] = z - drift_rev * self.dt + diffusion

        return zs
   
    
    def solve_pfode(self, zT, score_model):
        """ Solves the reverse-time PFODE for image data. """
        B, _, H, W = zT.shape
        zs = torch.zeros((B, self.n_steps, self.dim, H, W), device=self.device)
        zs[:, -1, :, :, :] = zT
        C = self.channel_dim

        ts = self.ts.to(self.device)
        dt_back = torch.empty(self.n_steps, device=self.device, dtype=ts.dtype)
        dt_back[0] = ts[0]
        dt_back[1:] = ts[1:] - ts[:-1]

        for i in range(self.n_steps - 1, -1, -1):
            z = zs[:, i, :, :, :]
            t_val = ts[i].expand(B)

            # --- Compute score from network ---
            ps_inputs = z[:, C:, :, :]
            score_ps = score_model(ps_inputs, t_val)
            score_full = torch.cat(
                [torch.zeros(B, C, H, W, device=self.device), score_ps], 
                dim=1
            )

            # --- Forward drift ---
            f_fwd = -self.beta * torch.einsum('ij,bcjhw->bcihw', self.A, z)

            # --- Deterministic ODE drift ---
            score_drift = torch.einsum('ij,bcjhw->bcihw', self.GGt, score_full)
            drift_ode = f_fwd - 0.5 * score_drift

            if i > 0:
                zs[:, i - 1, :, :, :] = z - drift_ode * dt_back[i]

        return zs


    def generate_samples(self, n_samples, img_shape, score_model, method='pfode'):
        """
        Generates samples by solving the reverse process.
        img_shape: (C, H, W)
        """
        zT = self.sample_prior(n_samples, img_shape)
        
        if method == 'pfode':
            paths = self.solve_pfode(zT, score_model)
        elif method == 'em':
            paths = self.solve_reverse_sde_em(zT, score_model)
        else:
            raise ValueError(f"Unknown generation method: {method}")
            
        # Return (B, n_steps, 3C, H, W)
        return paths

    
    def train_score_network(self, ScoreNetwork, dataloader, n_epochs=50, lr=1e-3, val_dataloader=None, **model_kwargs):
        """
        Trains the score network (UNet) using Denoising Score Matching
        on image-shaped data.
        """
        C = self.channel_dim
        # Pass model_kwargs (like in_channels) to the network constructor
        model = ScoreNetwork(**model_kwargs).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        GGt_ps = self.GGt[C:, C:] # (2C, 2C)

        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            model.train()
            total_train_loss = 0.0
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]")
            
            for x0_batch in pbar:
                # x0_batch is (B, C, H, W) from dataloader
                B, C_img, H, W = x0_batch.shape
                if B == 0: continue
                
                # 1. Sample z0 = [x0, p0, s0]
                x0 = x0_batch.to(self.device)
                p0 = torch.randn_like(x0) * np.sqrt(self.p_init_var)
                s0 = torch.randn_like(x0) * np.sqrt(self.s_init_var)
                z0 = torch.cat([x0, p0, s0], dim=1) # (B, 3C, H, W)

                # 2. Sample time index t
                t_idx = int(torch.randint(0, self.n_steps, (1,)).item())
                t_scalar = self.ts[t_idx]
                t_batch = t_scalar.expand(B).to(self.device)

                # 3. Get precomputed (3C, 3C) kernels
                M_t = self.M_t_all[t_idx]
                L_t = self.L_t_all[t_idx]
                L_t_trans = L_t.T

                # 4. Perturb: z_t = M_t z_0 + L_t eps
                eps = torch.randn_like(z0) # (B, 3C, H, W)
                
                # Apply (3C, 3C) matrices across channel dim of (B, 3C, H, W)
                mu_t = torch.einsum('ij,bcjhw->bcihw', M_t, z0)
                noise = torch.einsum('ij,bcjhw->bcihw', L_t, eps)
                z_t = mu_t + noise

                # 5. Target eps
                eps_target_ps = eps[:, C:, :, :] # (B, 2C, H, W)

                # 6. Prediction
                ps_inputs = z_t[:, C:, :, :] # (B, 2C, H, W)
                score_ps_pred = model(ps_inputs, t_batch) # (B, 2C, H, W)

                # 7. Reparametrize prediction to eps_hat
                score_full_pred = torch.cat(
                    [torch.zeros_like(x0), score_ps_pred], 
                    dim=1
                ) # (B, 3C, H, W)
                
                # eps_hat = - L_t^T @ score_pred
                eps_hat = -torch.einsum('ij,bcjhw->bcihw', L_t_trans, score_full_pred)
                eps_hat_ps = eps_hat[:, C:, :, :] # (B, 2C, H, W)

                # 8. Weighted loss with GGt, summed over pixels
                diff = eps_hat_ps - eps_target_ps
                
                # || diff ||^2_{GGt_ps}
                # (B, 2C, H, W), (2C, 2C), (B, 2C, H, W) -> (B, H, W)
                per_pixel_loss = torch.einsum("bcihw,ij,bcjhw->bhw", diff, GGt_ps, diff)
                
                # Sum over spatial dims and mean over batch
                loss = per_pixel_loss.sum(dim=(1, 2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                pbar.set_postfix(loss=loss.item() / (B * H * W)) # Show avg pixel loss
            
            avg_train_loss = total_train_loss / len(dataloader.dataset)
            train_losses.append(avg_train_loss)

            # --- Validation Loop (similar logic) ---
            if val_dataloader:
                model.eval()
                total_val_loss = 0.0
                pbar_val = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{n_epochs} [Val]")
                with torch.no_grad():
                    for x0_batch in pbar_val:
                        B, C_img, H, W = x0_batch.shape
                        if B == 0: continue
                        
                        x0 = x0_batch.to(self.device)
                        p0 = torch.randn_like(x0) * np.sqrt(self.p_init_var)
                        s0 = torch.randn_like(x0) * np.sqrt(self.s_init_var)
                        z0 = torch.cat([x0, p0, s0], dim=1) 
                        
                        t_idx = int(torch.randint(0, self.n_steps, (1,)).item())
                        t_scalar = self.ts[t_idx]
                        t_batch = t_scalar.expand(B).to(self.device)
                        
                        M_t = self.M_t_all[t_idx]
                        L_t = self.L_t_all[t_idx]
                        L_t_trans = L_t.T
                        
                        eps = torch.randn_like(z0)
                        mu_t = torch.einsum('ij,bcjhw->bcihw', M_t, z0)
                        noise = torch.einsum('ij,bcjhw->bcihw', L_t, eps)
                        z_t = mu_t + noise
                        
                        eps_target_ps = eps[:, C:, :, :]
                        
                        ps_inputs = z_t[:, C:, :, :]
                        score_ps_pred = model(ps_inputs, t_batch)
                        
                        score_full_pred = torch.cat(
                            [torch.zeros_like(x0), score_ps_pred], dim=1)
                        
                        eps_hat = -torch.einsum('ij,bcjhw->bcihw', L_t_trans, score_full_pred)
                        eps_hat_ps = eps_hat[:, C:, :, :]
                        
                        diff = eps_hat_ps - eps_target_ps
                        per_pixel_loss = torch.einsum("bcihw,ij,bcjhw->bhw", diff, GGt_ps, diff)
                        loss = per_pixel_loss.sum(dim=(1, 2)).mean()

                        total_val_loss += loss.item()
                        pbar_val.set_postfix(loss=loss.item() / (B * H * W))

                avg_val_loss = total_val_loss / len(val_dataloader.dataset)
                val_losses.append(avg_val_loss)
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            else:
                print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.6f}")

        return model, (train_losses, val_losses)