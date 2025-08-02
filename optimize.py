import numpy as np
from scipy.optimize import minimize
import warnings

def eigenvalue_objective_function(params):
    """
    Calculates the maximum real part of the eigenvalues of the drift matrix A.
    This optimizes for the slowest mode of the entire system.
    """
    gamma, lam, c = params
    A = - np.array([
        [0,           -1,                  0],
        [1,           gamma**2,            gamma * lam * c],
        [0,           gamma * lam * c,     lam**2]
    ])
    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.real(eigenvalues))

def frequency_objective_function(params, omega_range, num_points=100, beta=1.0):
    """
    Calculates the integrated response (susceptibility) of the system over a 
    specified frequency range. Minimizing this damps motion within that range.

    Args:
        params (list): A list containing the parameters [gamma, lambda, c].
        omega_range (tuple): A tuple of (omega_min, omega_max).
        num_points (int): Number of frequency points to evaluate in the range.
        beta (float): Inverse temperature scaling factor.

    Returns:
        float: The integrated squared norm of the susceptibility matrix.
    """
    gamma, lam, c = params
    omega_min, omega_max = omega_range

    # Define the 3x3 matrix A for a 1D harmonic oscillator (M=1, K=1)
    A = - np.array([
        [0,           -1,                  0],
        [1,           gamma**2,            gamma * lam * c],
        [0,           gamma * lam * c,     lam**2]
    ])
    
    omegas = np.linspace(omega_min, omega_max, num_points)
    identity = np.identity(3)
    total_response = 0.0

    with warnings.catch_warnings():        
        for omega in omegas:
            try:
                chi_matrix = np.linalg.inv((1j * omega * identity) + (beta * A))
                response_at_omega = np.linalg.norm(chi_matrix, 'fro')
                total_response += response_at_omega**2
            except np.linalg.LinAlgError:
                return 1e12

    return total_response

OPTIMIZATION_MODE = "eigenvalue" # "eigenvalue" or "frequency_range"
if OPTIMIZATION_MODE == "frequency_range":
  TARGET_OMEGA_RANGE = (5.0, 10.0) 
initial_guess = [1.0, 1.0, 0.5] # gamma, lambda, c
bounds = [(1e-9, None), (1e-9, None), (1e-9, 1.0 - 1e-9)]

if OPTIMIZATION_MODE == "eigenvalue":
    print("Optimizing for the slowest system mode (eigenvalue method)...")
    result = minimize(eigenvalue_objective_function, initial_guess, method='L-BFGS-B', bounds=bounds)
elif OPTIMIZATION_MODE == "frequency_range":
    print(f"Optimizing to damp frequency range: {TARGET_OMEGA_RANGE}...")
    objective_with_args = lambda params: frequency_objective_function(params, TARGET_OMEGA_RANGE)
    result = minimize(objective_with_args, initial_guess, method='L-BFGS-B', bounds=bounds)

if result.success:
    optimal_gamma, optimal_lambda, optimal_c = result.x
    
    print("\nOptimization Successful!")
    print("-------------------------")
    print(f"Optimal gamma: {optimal_gamma:.4f}")
    print(f"Optimal lambda: {optimal_lambda:.4f}")
    print(f"Optimal c: {optimal_c:.4f}")

    if OPTIMIZATION_MODE == "eigenvalue":
        print(f"\nMinimized maximum real part of eigenvalues: {result.fun:.4f}")
    else:
        print(f"\nMinimized integrated response in target range: {result.fun:.4f}")
else:
    print("\nOptimization failed.")
    print(result.message)