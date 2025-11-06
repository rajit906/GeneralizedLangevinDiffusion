# utils/viz.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def plot_losses(train_losses, val_losses, save_path):
    """Plots training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    if val_losses:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

def plot_gmm_comparison(x_real, x_fake, save_path):
    """Plots KDE of real vs. fake 1D data."""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x_real, label="Real Data (GMM)", color='blue', fill=True)
    sns.kdeplot(x_fake, label="Generated Samples", color='red', fill=True, alpha=0.7)
    plt.title("Real vs. Generated 1D Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()

# --- OLD 2D PLOT (can be removed) ---
def plot_gmm_comparison_2d(x_real, x_fake, save_path, title_suffix=""):
    # ... (this function is no longer called in the 2D case) ...
    pass

# --- NEW 1x3 COMBINED 2D DISTRIBUTION PLOT ---
def plot_gmm_comparison_2d_combined(x_real, x_fake_pfode, x_fake_sde, save_path):
    """Plots 2D KDE of real, PFODE, and SDE data in one figure."""
    # Set plot limits based on data
    x_min = x_real[:, 0].min() - 2
    x_max = x_real[:, 0].max() + 2
    y_min = x_real[:, 1].min() - 2
    y_max = x_real[:, 1].max() + 2
    limits = [min(x_min, y_min), max(x_max, y_max)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Real Data
    axs[0].set_title("Real Data Distribution")
    sns.kdeplot(x=x_real[:, 0], y=x_real[:, 1], 
                cmap="Blues", fill=True, thresh=0.05, ax=axs[0])
    axs[0].set_xlabel("$x_0$"); axs[0].set_ylabel("$x_1$")
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_xlim(limits); axs[0].set_ylim(limits)

    # Plot PFODE Data
    axs[1].set_title("Generated (PFODE)")
    sns.kdeplot(x=x_fake_pfode[:, 0], y=x_fake_pfode[:, 1], 
                cmap="Reds", fill=True, thresh=0.05, ax=axs[1])
    axs[1].set_xlabel("$x_0$"); axs[1].set_ylabel("$x_1$")
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_xlim(limits); axs[1].set_ylim(limits)

    # Plot SDE Data
    axs[2].set_title("Generated (SDE)")
    sns.kdeplot(x=x_fake_sde[:, 0], y=x_fake_sde[:, 1], 
                cmap="Greens", fill=True, thresh=0.05, ax=axs[2])
    axs[2].set_xlabel("$x_0$"); axs[2].set_ylabel("$x_1$")
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_xlim(limits); axs[2].set_ylim(limits)

    fig.suptitle("Real vs. Generated 2D Distributions")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_1d_trajectories(paths, ts, data_dim, save_path):
    """
    Plots 1D trajectories (vs. time) for x, p, and s for the first dimension.
    paths shape: (B, n_steps, 3*d)
    """
    # ... (function contents are unchanged) ...
    B, n_steps, _ = paths.shape
    paths_np = paths.cpu().numpy()
    ts_np = ts
    x_paths = paths_np[:, :, 0]
    p_paths = paths_np[:, :, data_dim]
    s_paths = paths_np[:, :, 2*data_dim]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].plot(ts_np, x_paths.T, color='blue', alpha=0.1)
    axs[0].set_title("Position $x_0(t)$"); axs[0].set_xlabel("Time t"); axs[0].set_ylabel("$x_0$")
    axs[1].plot(ts_np, p_paths.T, color='green', alpha=0.1)
    axs[1].set_title("Momentum $p_0(t)$"); axs[1].set_xlabel("Time t"); axs[1].set_ylabel("$p_0$")
    axs[2].plot(ts_np, s_paths.T, color='red', alpha=0.1)
    axs[2].set_title("Auxiliary $s_0(t)$"); axs[2].set_xlabel("Time t"); axs[2].set_ylabel("$s_0$")
    fig.suptitle(f"1D Trajectories (First Dimension, {B} paths)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- OLD 2D TRAJECTORY PLOT (can be removed) ---
def plot_2d_trajectories(paths, title_suffix, save_path):
    # ... (this function is no longer called in the 2D case) ...
    pass

# --- NEW 1x2 COMBINED 2D TRAJECTORY PLOT ---
def plot_2d_trajectories_combined(paths_pfode, paths_sde, save_path):
    """
    Plots 2D position (x_1 vs x_0) trajectories for PFODE and SDE side-by-side.
    paths shape: (B, n_steps, 3*d)
    """
    B_pfode, _, _ = paths_pfode.shape
    B_sde, _, _ = paths_sde.shape
    
    pfode_np = paths_pfode.cpu().numpy()
    sde_np = paths_sde.cpu().numpy()
    
    # Extract 1st and 2nd dim of x
    x_pfode_0 = pfode_np[:, :, 0] # (B, n_steps)
    x_pfode_1 = pfode_np[:, :, 1] # (B, n_steps)
    x_sde_0 = sde_np[:, :, 0]
    x_sde_1 = sde_np[:, :, 1]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot PFODE Trajectories
    for i in range(B_pfode):
        axs[0].plot(x_pfode_0[i], x_pfode_1[i], color='blue', alpha=0.1)
    axs[0].scatter(x_pfode_0[:, -1], x_pfode_1[:, -1], color='red', s=10, alpha=0.5, label='Start (t=T, Noise)')
    axs[0].scatter(x_pfode_0[:, 0], x_pfode_1[:, 0], color='green', s=10, alpha=0.5, label='End (t=0, Data)')
    axs[0].set_title("2D Position Trajectories (PFODE)")
    axs[0].set_xlabel("$x_0$"); axs[0].set_ylabel("$x_1$")
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].legend()
    
    # Plot SDE Trajectories
    for i in range(B_sde):
        axs[1].plot(x_sde_0[i], x_sde_1[i], color='green', alpha=0.1)
    axs[1].scatter(x_sde_0[:, -1], x_sde_1[:, -1], color='red', s=10, alpha=0.5, label='Start (t=T, Noise)')
    axs[1].scatter(x_sde_0[:, 0], x_sde_1[:, 0], color='blue', s=10, alpha=0.5, label='End (t=0, Data)')
    axs[1].set_title("2D Position Trajectories (SDE)")
    axs[1].set_xlabel("$x_0$"); axs[1].set_ylabel("$x_1$")
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# --- OLD ANIMATION (can be removed) ---
def create_animation(paths, ts, data_dim, title_suffix, save_path, 
                     subsample_factor=10, n_samples_to_plot=500):
    # ... (this function is no longer called in the 2D case) ...
    pass

# --- NEW 1x2 COMBINED ANIMATION ---
def create_animation_combined(paths_pfode, paths_sde, ts, data_dim, save_path, 
                              subsample_factor=10, n_samples_to_plot=500):
    """
    Creates a GIF animation of the reverse process (noise-to-data)
    with PFODE and SDE side-by-side.
    """
    
    # Subsample for speed
    paths_pfode_sub = paths_pfode[:n_samples_to_plot, ::subsample_factor, :data_dim]
    paths_sde_sub = paths_sde[:n_samples_to_plot, ::subsample_factor, :data_dim]
    ts_sub = ts[::subsample_factor]
    
    # Permute to (T, B, D)
    pfode_np = paths_pfode_sub.permute(1, 0, 2).cpu().numpy()
    sde_np = paths_sde_sub.permute(1, 0, 2).cpu().numpy()
    ts_np = ts_sub.cpu().numpy()
    
    n_frames = pfode_np.shape[0]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Get global data limits for stable axes
    x_min = min(pfode_np[..., 0].min(), sde_np[..., 0].min()) - 1
    x_max = max(pfode_np[..., 0].max(), sde_np[..., 0].max()) + 1
    y_min = min(pfode_np[..., 1].min(), sde_np[..., 1].min()) - 1
    y_max = max(pfode_np[..., 1].max(), sde_np[..., 1].max()) + 1
    limits = [min(x_min, y_min), max(x_max, y_max)]

    def update(frame):
        # frame goes from 0 to n_frames-1
        # We want to plot from noise (last frame) to data (first frame)
        i = (n_frames - 1) - frame
        
        axs[0].clear()
        axs[1].clear()
        
        data_pfode = pfode_np[i, :, :] # (B, D)
        data_sde = sde_np[i, :, :]   # (B, D)
        
        # Plot PFODE
        axs[0].scatter(data_pfode[:, 0], data_pfode[:, 1], alpha=0.3, s=10, color='blue')
        axs[0].set_title(f"PFODE - Time t = {ts_np[i]:.3f}")
        axs[0].set_xlabel("$x_0$"); axs[0].set_ylabel("$x_1$")
        axs[0].set_xlim(limits); axs[0].set_ylim(limits)
        axs[0].set_aspect('equal', adjustable='box')

        # Plot SDE
        axs[1].scatter(data_sde[:, 0], data_sde[:, 1], alpha=0.3, s=10, color='green')
        axs[1].set_title(f"Reverse SDE (EM) - Time t = {ts_np[i]:.3f}")
        axs[1].set_xlabel("$x_0$"); axs[1].set_ylabel("$x_1$")
        axs[1].set_xlim(limits); axs[1].set_ylim(limits)
        axs[1].set_aspect('equal', adjustable='box')

    # Create animation
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100)
    
    # Save as GIF
    anim.save(save_path, writer='pillow', fps=15)
    plt.close(fig)