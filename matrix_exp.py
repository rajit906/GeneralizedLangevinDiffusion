import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

def compute_mean_and_covariance(t, beta, A, G, mu_0, Sigma_0):
    """
    Computes the mean and covariance for a linear stochastic system at time t.
    
    The system is defined by dx = (-beta * A) * x * dt + G * dW.
    
    Args:
        t (float): The target time.
        beta (float): The beta parameter.
        A (np.ndarray): The drift matrix component.
        G (np.ndarray): The diffusion coefficient matrix.
        mu_0 (np.ndarray): The initial mean vector at t=0.
        Sigma_0 (np.ndarray): The initial covariance matrix at t=0.
        
    Returns:
        (np.ndarray, np.ndarray): A tuple containing the mean vector and covariance matrix at time t.
    """
    # Define the core drift and diffusion matrices
    F = -beta * A
    Q = G @ G.T
    
    # --- 1. Compute the Mean ---
    M_t = expm(F * t)
    mu_t = M_t @ mu_0
    
    # --- 2. Compute the Covariance ---
    # Propagated term
    propagated_term = M_t @ Sigma_0 @ M_t.T
    
    # Integral term using the corrected formula
    def integrand(s):
        M_s = expm(F * s)
        return M_s @ Q @ M_s.T

    # Numerically integrate each element of the matrix from 0 to t
    integral_term = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            element_integrand = lambda s: integrand(s)[i, j]
            result, _ = quad(element_integrand, 0, t, epsabs=1e-8, epsrel=1e-8)
            integral_term[i, j] = result
            
    Sigma_t = propagated_term + integral_term
    
    return mu_t, Sigma_t

# --- Analytical Test Case ---
if __name__ == '__main__':
    print("--- Analytical Test: A Nilpotent System ---")
    
    # 1. SETUP THE TEST CASE
    # We choose parameters that make F = -beta*A nilpotent (F^2 = 0).
    # This simplifies exp(F*s) to (I + F*s), making the integral easy to solve by hand.
    target_time = 5.0
    beta = 1.0
    gamma = 2.0  # Let's use gamma=2 to make it more interesting
    
    # Parameters for A and G to create the nilpotent case
    A_test = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
    G_test = np.array([[0., 0., 0.], [0., gamma, 0.], [0., 0., 0.]])
    
    # Initial conditions
    mu_0_test = np.array([10.0, 20.0, 30.0])
    Sigma_0_test = np.zeros((3, 3)) # Start with zero uncertainty
    
    # 2. RUN THE GENERAL-PURPOSE FUNCTION
    print(f"Running general function for t = {target_time}...")
    mu_computed, Sigma_computed = compute_mean_and_covariance(
        target_time, beta, A_test, G_test, mu_0_test, Sigma_0_test
    )

    # 3. PERFORM THE DIRECT ANALYTICAL CALCULATION
    # print("Calculating the exact analytical solution for this specific case...")
    # F_test = -beta * A_test # -> F_test[1,0] = -1.0
    
    # Analytical Mean: mu(t) = exp(F*t)*mu_0 = (I + F*t)*mu_0
    # mu_analytical = np.array([
    #     mu_0_test[0],
    #     mu_0_test[1] + (F_test[1,0] * target_time * mu_0_test[0]),
    #     mu_0_test[2]
    # ])
    
    # # Analytical Covariance: Integral of M(s)Q M(s)^T ds
    # # For this specific case, the integrand simplifies to a constant matrix Q.
    # # So the integral is simply Q * t.
    # Q_test = G_test @ G_test.T # Q[1,1] will be gamma^2 = 4
    # Sigma_analytical = Q_test * target_time

    # # 4. COMPARE THE RESULTS
    # print("\n--- COMPARISON ---")
    # np.set_printoptions(precision=5, suppress=True)
    
    # print("\nMean Vector μ(t):")
    # print("Computed:  ", mu_computed)
    # print("Analytical:", mu_analytical)
    
    # print("\nCovariance Matrix Σ(t):")
    # print("Computed:\n", Sigma_computed)
    # print("Analytical:\n", Sigma_analytical)

    # # Verification check
    # mean_match = np.allclose(mu_computed, mu_analytical)
    # cov_match = np.allclose(Sigma_computed, Sigma_analytical)
    
    # print("\nVerification:")
    # print(f"  Mean matches analytical result: {mean_match} {'✅' if mean_match else '❌'}")
    # print(f"  Covariance matches analytical result: {cov_match} {'✅' if cov_match else '❌'}")