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