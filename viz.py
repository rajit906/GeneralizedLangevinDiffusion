import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
# --- Plotting Helper Functions ---
def plot_position_dist(generated_samples, gmm_params, ax):
    """Plots the final position distribution against the true GMM PDF."""
    x_range = np.linspace(-8, 8, 400)
    true_pdf = np.zeros_like(x_range)
    for w, m, s in zip(gmm_params['weights'], gmm_params['means'], gmm_params['stds']):
        true_pdf += w.cpu().numpy() * (1 / (s.cpu().numpy() * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - m.cpu().numpy()) / s.cpu().numpy())**2)
    
    ax.plot(x_range, true_pdf, 'r--', lw=2, label=r'Target $p_0(x)$')
    ax.hist(generated_samples, bins=60, density=True, alpha=0.75, label='Generated Samples')
    ax.set_title('Final Position Distribution'); ax.set_xlabel('Position x'); ax.set_ylabel('Density'); ax.legend()

def plot_aux_dist(ax, *args):
    """Plots one or more auxiliary distributions (e.g., momentum) on the same axes."""
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(args)))
    for i, (data, label) in enumerate(args):
        ax.hist(data, bins=60, density=True, alpha=0.6, label=f'Final {label}', color=colors[i])
    ax.set_title(f'Final Auxiliary Distribution(s)'); ax.set_xlabel('Value'); ax.set_ylabel('Density'); ax.legend()