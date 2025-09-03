import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov
from scipy.integrate import solve_ivp

def stationary_covariance(beta, A, G):
    """Solve F C + C F^T + Q = 0 for C."""
    F = -beta * A
    Q = G @ G.T
    return solve_continuous_lyapunov(F, -Q)

def compute_mean_and_covariance(t, beta, A, G, mu_0, Sigma_0, C):
    """Analytical covariance using stationary solution C."""
    F = -beta * A
    M_t = expm(F * t)
    mu_t = M_t @ mu_0
    Sigma_t = C + M_t @ (Sigma_0 - C) @ M_t.T
    return mu_t, Sigma_t

# ---------------- TESTS ----------------
def test_pure_diffusion():
    """If A=0, then Sigma(t) = Sigma_0 + (G G^T) * t"""
    A = np.zeros((2, 2))
    G = np.eye(2)
    mu0 = np.zeros(2)
    Sigma0 = np.zeros((2, 2))
    t = 1.0
    beta = 1.0
    C = stationary_covariance(beta, A, G)
    mu, Sigma = compute_mean_and_covariance(t, beta, A, G, mu0, Sigma0, C)
    expected = np.eye(2) * t
    assert np.allclose(Sigma, expected, atol=1e-5), f"Expected {expected}, got {Sigma}"

def test_stationary_covariance():
    """For OU: dX = -X dt + sqrt(2) dW, stationary variance should be 1"""
    A = np.array([[1.0]])
    G = np.array([[np.sqrt(2.0)]])
    mu0 = np.zeros(1)
    Sigma0 = np.zeros((1, 1))
    beta = 1.0
    t = 10.0
    C = stationary_covariance(beta, A, G)
    mu, Sigma = compute_mean_and_covariance(t, beta, A, G, mu0, Sigma0, C)
    # Long time covariance should converge to 1
    assert np.allclose(Sigma, np.array([[1.0]]), atol=1e-2), f"Expected ~1, got {Sigma}"

def test_mean_propagation():
    """Check mean evolves correctly for OU system"""
    A = np.array([[1.0]])
    G = np.array([[np.sqrt(2.0)]])
    mu0 = np.array([1.0])
    Sigma0 = np.zeros((1, 1))
    beta = 1.0
    t = 1.0
    C = stationary_covariance(beta, A, G)
    mu, Sigma = compute_mean_and_covariance(t, beta, A, G, mu0, Sigma0, C)
    expected_mu = np.exp(-t) * mu0  # exact OU mean
    assert np.allclose(mu, expected_mu, atol=1e-5), f"Expected {expected_mu}, got {mu}"

def test_gld_vs_cld_consistency():
    """
    Check that GLD reduces to CLD when λ=c=0.
    The first two coordinates of GLD (x,v) should match the CLD (x,v).
    """
    t = 0.5
    beta = 1.0

    # Critically damped Langevin (2D: position + velocity)
    A_cld = np.array([[0.0, -1.0],
                      [1.0,  1.0]])
    G_cld = np.array([[0.0],
                      [1.0]])
    mu0_cld = np.array([-4.0, 0.0])
    Sigma0_cld = np.eye(2)

    mu_cld, Sigma_cld = compute_mean_and_covariance(t, beta, A_cld, G_cld, mu0_cld, Sigma0_cld)

    # Generalized Langevin Diffusion (3D: position + velocity + auxiliary var)
    # Set λ=c=0 → extra dimension decouples
    A_gld = np.array([[0.0, -1.0,  0.0],
                      [1.0,  1.0,  0.0],
                      [0.0,  0.0,  0.0]])
    G_gld = np.array([[0.0],
                      [1.0],
                      [0.0]])
    mu0_gld = np.array([-4.0, 0.0, 0.0])
    Sigma0_gld = np.eye(3)

    mu_gld, Sigma_gld = compute_mean_and_covariance(t, beta, A_gld, G_gld, mu0_gld, Sigma0_gld)

    # Extract 2D marginal from GLD
    mu_gld_marg = mu_gld[:2]
    Sigma_gld_marg = Sigma_gld[:2, :2]

    # Compare
    print("CLD mean:", mu_cld)
    print("GLD marginal mean:", mu_gld_marg)
    print("CLD cov:\n", Sigma_cld)
    print("GLD marginal cov:\n", Sigma_gld_marg)

    assert np.allclose(mu_cld, mu_gld_marg, atol=1e-6)
    assert np.allclose(Sigma_cld, Sigma_gld_marg, atol=1e-6)


# # Run tests
#test_pure_diffusion()
# test_stationary_covariance()
# test_mean_propagation()
# #test_gld_vs_cld_consistency()
# print("All sanity checks passed ✅")


# def compute_covariance_ode(t, beta, A, G, Sigma_0):
#     """
#     Computes Sigma(t) by solving the Lyapunov differential equation.
#     This is often more numerically stable.
#     """
#     n = A.shape[0]
#     F = -beta * A
#     Q = G @ G.T

#     # The ODE function needs a flattened state vector
#     def lyapunov_ode(t, sigma_flat):
#         Sigma = sigma_flat.reshape(n, n)
#         dSigma_dt = F @ Sigma + Sigma @ F.T + Q
#         return dSigma_dt.flatten()

#     sigma0_flat = Sigma_0.flatten()
#     sol = solve_ivp(
#         lyapunov_ode, 
#         [0, t], 
#         sigma0_flat, 
#         t_eval=[t]
#     )
    
#     Sigma_t = sol.y[:, -1].reshape(n, n)
#     return Sigma_t

# def compute_mean_and_covariance(t, beta, A, G, mu_0, Sigma_0):
#     """
#     Computes the mean and covariance for a linear stochastic system at time t.
#     dx = (-beta * A) x dt + G dW
#     """
#     F = -beta * A
#     Q = G @ G.T

#     # Mean propagation
#     M_t = expm(F * t)
#     mu_t = M_t @ mu_0

#     Sigma_t = compute_covariance_ode(t, beta, A, G, Sigma_0)
#     return mu_t, Sigma_t