# viz.py
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.style.use('seaborn-v0_8-whitegrid')
# --- Plotting Helper Functions ---
def plot_position_dist(generated_samples, gmm_params, ax):
    """Plots the final position distribution against the true GMM PDF."""
    # Ensure NumPy
    if isinstance(generated_samples, torch.Tensor):
        samples_np = generated_samples.detach().cpu().numpy()
    else:
        samples_np = np.asarray(generated_samples)

    pad = 2.0
    x_min, x_max = samples_np.min(), samples_np.max()
    x_range = np.linspace(x_min - pad, x_max + pad, 400)

    true_pdf = np.zeros_like(x_range)
    for w, m, s in zip(gmm_params['weights'], gmm_params['means'], gmm_params['stds']):
        w, m, s = float(w), float(m), float(s)
        true_pdf += w * (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - m) / s) ** 2)

    ax.hist(samples_np, bins=100, density=True, alpha=0.75, label="Generated Samples")
    ax.plot(x_range, true_pdf, "r--", lw=2, label=r"Target $p_0(x)$")
    ax.set_title("Final Position Distribution")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Density")
    ax.legend()



def plot_aux_dist(ax, *args, target_dist=None):
    """Plots one or more auxiliary distributions (e.g., momentum) on the same axes."""
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(args)))
    for i, (data, label) in enumerate(args):
        ax.hist(data, bins=100, density=True, alpha=0.6, label=f'Final {label}', color=colors[i])
    if target_dist is not None:
        mean, std = target_dist
        x_min, x_max = ax.get_xlim()
        x_range = np.linspace(x_min, x_max, 200)
        pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)
        ax.plot(x_range, pdf, 'r--', lw=2, label='Target PDF')
    ax.set_title(f'Final Auxiliary Distribution(s)'); ax.set_xlabel('Value'); ax.set_ylabel('Density'); ax.legend()