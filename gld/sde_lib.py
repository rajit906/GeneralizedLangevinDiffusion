# sde_lib.py
# TODO: Store Kronecker products for A.
import torch
import numpy as np
from tqdm import tqdm
import scipy.linalg
from gmm.matrix_exp import stationary_covariance, compute_mean_and_covariance 

class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=20, N=1000, device='cuda'):
        """
        Construct a Variance Preserving SDE.
        
        Args:
            beta_min: a `float` for the minimum beta variance.
            beta_max: a `float` for the maximum beta variance.
            N: an `int` number of discretization steps.
            device: 'cuda' or 'cpu'
        """
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.device = device
        self.t_span = torch.linspace(1e-5, 1.0, N, device=device)

        # Precompute alphas and betas
        self.betas = torch.linspace(beta_min, beta_max, N, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def marginal_prob(self, x_0, t):
        """
        Compute the mean and standard deviation of the perturbation kernel q(x_t | x_0).
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff[:, None]) * x_0
        std = torch.sqrt(1.0 - torch.exp(2. * log_mean_coeff))
        return mean, std[:, None]

    def sde(self, x, t):
        """
        Compute the drift and diffusion coefficients for the SDE.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    @torch.no_grad()
    def reverse_sde_sampler(self, model, shape, steps):
        """
        Sample from the reverse-time SDE using the Euler-Maruyama method.
        """
        x = torch.randn(shape, device=self.device)
        dt = 1. / steps
        
        for t_val in np.linspace(1, 1e-5, steps):
            t = torch.ones(shape[0], device=self.device) * t_val
            
            # Predict noise and convert to score
            predicted_noise = model(x, t)
            _, std = self.marginal_prob(torch.zeros_like(x), t)
            score = -predicted_noise / std
            
            # SDE update
            drift, diffusion = self.sde(x, t)
            drift = drift - diffusion[:, None]**2 * score
            z = torch.randn_like(x)
            x = x - drift * dt + diffusion[:, None] * torch.sqrt(torch.tensor(dt)) * z
            
        return x

    @torch.no_grad()
    def ode_sampler(self, model, shape, steps):
        """
        Sample from the Probability Flow ODE.
        """
        x = torch.randn(shape, device=self.device)
        dt = 1. / steps

        for t_val in np.linspace(1, 1e-5, steps):
            t = torch.ones(shape[0], device=self.device) * t_val
            
            # Predict noise and convert to score
            predicted_noise = model(x, t)
            _, std = self.marginal_prob(torch.zeros_like(x), t)
            score = -predicted_noise / std

            # ODE update
            drift, diffusion = self.sde(x, t)
            drift = drift - 0.5 * diffusion[:, None]**2 * score
            x = x - drift * dt

        return x

class GLDSDE:
    """
    Implements the Generalized Langevin Diffusion SDE for N-dimensional data.
    The state vector is z = [x, p, s], where each is N-dimensional.
    The total state dimension is 3*N.
    """
    def __init__(self, 
                 data_dim=2, 
                 T=1.0, 
                 n_steps=1000, 
                 gamma=1.0,
                 lambda_val=1.0, 
                 c_val = 0.1, 
                 M=1., 
                 beta=8.0,
                 p_init_var=1.0,
                 s_init_var=1.0,
                 device='cuda'):
        
        print(f"--- Initializing GLDSDE (Data Dim: {data_dim}) ---")
        self.data_dim = data_dim
        self.state_dim = 3 * data_dim
        self.device = device

        # --- Time Discretization ---
        self.T = T
        self.N = n_steps # Use N for consistency with VPSDE
        self.ts = torch.linspace(1e-4, T, n_steps, device=device)
        self.dt = torch.tensor(T / n_steps, device=device, dtype=torch.float32)

        # --- Model Parameters ---
        self.gamma = gamma
        self.c = c_val
        self.lambda_val = lambda_val
        self.M = M
        self.M_inv = 1. / self.M
        self.beta = beta
        self.p_init_var = p_init_var
        self.s_init_var = s_init_var

        # --- Build 6x6 System Matrices ---
        I = torch.eye(data_dim, device=device, dtype=torch.float32)
        O = torch.zeros((data_dim, data_dim), device=device, dtype=torch.float32)

        # Hamiltonian part
        AH = torch.cat([
            torch.cat([O, -I * self.M_inv, O], dim=1),
            torch.cat([I, O, O], dim=1),
            torch.cat([O, O, O], dim=1)
        ], dim=0)

        # Damping part
        AGamma = torch.cat([
            torch.cat([O, O, O], dim=1),
            torch.cat([O, I * (self.M_inv * self.gamma**2), I * (self.gamma * self.lambda_val * self.c)], dim=1),
            torch.cat([O, I * (self.gamma * self.lambda_val * self.c), I * (self.lambda_val**2)], dim=1)
        ], dim=0)
        
        self.A = (AH + AGamma).to(device)

        # Noise matrix B
        B = torch.cat([
            torch.cat([O, O, O], dim=1),
            torch.cat([O, I * self.gamma, O], dim=1),
            torch.cat([O, I * (self.lambda_val * self.c), I * (self.lambda_val * np.sqrt(1 - self.c**2))], dim=1)
        ], dim=0)

        self.G = (np.sqrt(2 * self.beta) * B).to(device)
        self.GGt = (self.G @ self.G.T).to(device)
        
        # (p,s) block of GGt, needed for loss
        self.GGt_ps = self.GGt[self.data_dim:, self.data_dim:].to(device)

        # --- Precompute Stationary Covariance (on CPU) ---
        self.A_np = self.A.cpu().numpy()
        self.G_np = self.G.cpu().numpy()
        
        print("Calculating stationary covariance...")
        self.C_np = stationary_covariance(self.beta, self.A_np, self.G_np)
        self.C = torch.from_numpy(self.C_np).float().to(device)
        print("GLDSDE Initialized.")
        
        # Caches for marginal_prob
        self._F_t_cache = {}
        self._L_t_cache = {}

    def get_z0(self, x0):
        """Constructs z0 = [x0, p0, s0] from data x0."""
        B, D = x0.shape
        assert D == self.data_dim, "Data dim mismatch"
        
        p0 = torch.randn(B, D, device=self.device) * np.sqrt(self.p_init_var)
        s0 = torch.randn(B, D, device=self.device) * np.sqrt(self.s_init_var)
        z0 = torch.cat([x0, p0, s0], dim=1) # (B, 6)
        return z0

    def _get_kernel_params(self, t):
        """
        Computes and caches F(t) = expm(-beta*A*t) and L(t)
        where Sigma(t) = C - F(t)C F(t)^T = L(t)L(t)^T
        Note: t is a SCALAR.
        """
        t = float(t)
        if t not in self._F_t_cache:
            # Compute on CPU with numpy
            F_t_np = scipy.linalg.expm(-self.beta * self.A_np * t)
            Sigma_t_np = self.C_np - F_t_np @ self.C_np @ F_t_np.T
            
            # Add jitter for numerical stability
            jitter = 1e-6 * np.eye(self.state_dim)
            L_t_np = np.linalg.cholesky(Sigma_t_np + jitter)
            
            # Store as torch tensors on device
            self._F_t_cache[t] = torch.from_numpy(F_t_np).float().to(self.device)
            self._L_t_cache[t] = torch.from_numpy(L_t_np).float().to(self.device)
            
        return self._F_t_cache[t], self._L_t_cache[t]

    def marginal_prob(self, z0, t):
        """
        Computes the mean and Cholesky of the perturbation kernel p(z_t | z_0).
        t is a SCALAR.
        Returns:
            mu_t: (B, state_dim)
            L_t: (state_dim, state_dim)
        """
        F_t, L_t = self._get_kernel_params(t)
        mu_t = (F_t @ z0.T).T
        return mu_t, L_t

    def sde(self, z, t):
        """
        Returns the drift and noise matrices for the forward SDE.
        t is a batch (B,)
        """
        # A is (D,D), z is (B,D) -> (A @ z.T).T is (B,D)
        drift = -self.beta * (self.A @ z.T).T 
        return drift, self.G, self.GGt

    @torch.no_grad()
    def reverse_sde_sampler(self, model, shape, steps):
        """
        Sample from the reverse-time SDE using Euler-Maruyama.
        'shape' is the shape of the data x, e.g., (B, 2).
        """
        B, D = shape
        assert D == self.data_dim
        state_dim = self.state_dim

        # 1. Initialize zT from prior
        xT = torch.randn(B, D, device=self.device)
        pT = torch.randn(B, D, device=self.device) * np.sqrt(self.M)
        sT = torch.randn(B, D, device=self.device) * np.sqrt(self.s_init_var)
        z = torch.cat([xT, pT, sT], dim=1) # (B, 6)
        
        sqrt_dt = torch.sqrt(self.dt)

        for i in tqdm(range(self.N - 1, -1, -1), desc="GLD Reverse SDE Sampler"):
            t_idx = i
            t_scalar = self.ts[t_idx]
            t_batch = torch.full((B,), t_scalar, device=self.device)

            # 2. Get score from model
            ps_inputs = z[:, self.data_dim:] # (B, 4)
            score_ps = model(ps_inputs, t_batch) # (B, 4)
            score_full = torch.cat([torch.zeros(B, D, device=self.device), score_ps], dim=1) # (B, 6)

            # 3. Get SDE components
            f_fwd, G, GGt = self.sde(z, t_batch)
            
            # 4. Reverse drift
            score_drift = (GGt @ score_full.T).T
            drift_rev = f_fwd - score_drift
            
            # 5. Noise term
            dW = torch.randn_like(z) * sqrt_dt
            diffusion = (G @ dW.T).T
            
            # 6. Euler-Maruyama step (backward)
            z = z - drift_rev * self.dt + diffusion

        # Return only the x component
        x0 = z[:, :self.data_dim]
        return x0

    @torch.no_grad()
    def ode_sampler(self, model, shape, steps):
        """
        Sample from the Probability Flow ODE.
        'shape' is the shape of the data x, e.g., (B, 2).
        """
        B, D = shape
        assert D == self.data_dim
        state_dim = self.state_dim

        # 1. Initialize zT from prior
        xT = torch.randn(B, D, device=self.device)
        pT = torch.randn(B, D, device=self.device) * np.sqrt(self.M)
        sT = torch.randn(B, D, device=self.device) * np.sqrt(self.s_init_var)
        z = torch.cat([xT, pT, sT], dim=1) # (B, 6)

        for i in tqdm(range(self.N - 1, -1, -1), desc="GLD ODE Sampler"):
            t_idx = i
            t_scalar = self.ts[t_idx]
            t_batch = torch.full((B,), t_scalar, device=self.device)

            # 2. Get score from model
            ps_inputs = z[:, self.data_dim:] # (B, 4)
            score_ps = model(ps_inputs, t_batch) # (B, 4)
            score_full = torch.cat([torch.zeros(B, D, device=self.device), score_ps], dim=1) # (B, 6)

            # 3. Get SDE components
            f_fwd, _, GGt = self.sde(z, t_batch)
            
            # 4. ODE drift
            score_drift = (GGt @ score_full.T).T
            drift_ode = f_fwd - 0.5 * score_drift
            
            # 5. Euler-Maruyama step (backward, deterministic)
            z = z - drift_ode * self.dt

        # Return only the x component
        x0 = z[:, :self.data_dim]
        return x0