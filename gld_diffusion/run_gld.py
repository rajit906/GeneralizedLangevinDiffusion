# run_gld.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from diffusion.gld import GeneralizedLangevinDiffusion
from models import GLDScoreNetwork
from utils.data import GMMDataset
from utils.viz import (
    plot_losses, 
    plot_gmm_comparison, 
    plot_1d_trajectories, 
    plot_gmm_comparison_2d_combined,
    plot_2d_trajectories_combined,
    create_animation_combined
)

# Use cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Configuration ---
CONFIG = {
    # Data
    'data_dim': 2,
    'gmm_params': {
        "weights": [0.25, 0.25, 0.25, 0.25], 
        "means": [ [-5., -5.], [-5., 5.], [5., -5.], [5., 5.] ], 
        "stds": [ 1., 1., 1., 1. ]
    },
    # Training
    'do_train': True,
    'n_epochs': 25,
    'batch_size': 256,
    'lr': 1e-3,
    'n_train_samples': 50000, 
    'n_val_samples': 5000,
    # Diffusion
    'n_steps': 500,
    'T': 1.0,
    'gamma': 1.0,
    'lambda_val': 1.0,
    'c_val': 0.1,
    'M': 1.0,
    # Evaluation
    'n_eval_samples': 5000,
}
# --- 2. Setup Directories ---
subfolder_name = (
    f"dim{CONFIG['data_dim']}_gamma{CONFIG['gamma']}_lambda{CONFIG['lambda_val']}_"
    f"c{CONFIG['c_val']}_M{CONFIG['M']}_lr{CONFIG['lr']}_bs{CONFIG['batch_size']}"
)
save_dir = f"checkpoints/gld_general/{subfolder_name}"
plot_dir = os.path.join(save_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, "score_net_final.pt")

# --- 3. Load Data ---
print("Loading data...")
train_dataset = GMMDataset(CONFIG['gmm_params'], n_samples=CONFIG['n_train_samples'])
val_dataset = GMMDataset(CONFIG['gmm_params'], n_samples=CONFIG['n_val_samples'])

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=0 # Set to 0 for simplicity/compatibility
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False
)
print(f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")


# --- 4. Initialize Model & Precompute ---
print("Initializing model...")
gld = GeneralizedLangevinDiffusion(
    data_dim=CONFIG['data_dim'],
    n_steps=CONFIG['n_steps'],
    T=CONFIG['T'],
    inttype='em',
    gamma=CONFIG['gamma'],
    lambda_val=CONFIG['lambda_val'],
    c_val=CONFIG['c_val'],
    M=CONFIG['M'],
    device=DEVICE
)

# This is a new, crucial step
gld.precompute()

# --- 5. Training ---
if CONFIG['do_train']:
    print("Starting training...")
    model, (train_losses, val_losses) = gld.train_score_network(
        ScoreNetwork=GLDScoreNetwork,
        dataloader=train_loader,
        val_dataloader=val_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr']
    )

    # Save model and plots
    torch.save(model.state_dict(), checkpoint_path)
    plot_losses(
        train_losses, 
        val_losses, 
        os.path.join(plot_dir, "training_loss.png")
    )
    print(f"✅ Training complete. Model + loss plot saved in: {save_dir}")
else:
    print("Skipping training, loading model...")

# --- 6. Load Model for Evaluation ---
model = GLDScoreNetwork(data_dim=CONFIG['data_dim']).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
print("✅ Checkpoint loaded successfully.")

# --- 7. Evaluation & Plotting (HEAVILY MODIFIED) ---
print("Generating PFODE samples...")
with torch.no_grad():
    pfode_paths = gld.generate_samples(
        CONFIG['n_eval_samples'], 
        model, 
        method='pfode'
    ) # (B, n_steps, 3*d)

print("Generating Reverse SDE (EM) samples...")
with torch.no_grad():
    reverse_sde_paths = gld.generate_samples(
        CONFIG['n_eval_samples'],
        model,
        method='em'
    ) # (B, n_steps, 3*d)

# Extract generated x0 samples
x0_fake_pfode = pfode_paths[:, 0, :CONFIG['data_dim']].cpu().numpy()
x0_fake_sde = reverse_sde_paths[:, 0, :CONFIG['data_dim']].cpu().numpy()

# Get real data for comparison
x0_real = val_dataset.get_all_samples().cpu().numpy()


# --- Plot 1: Final Distribution Comparison ---
print("Plotting final distributions...")
if CONFIG['data_dim'] == 1:
    # 1D: Plot PFODE and SDE results separately
    plot_gmm_comparison(
        x0_real.flatten(), 
        x0_fake_pfode.flatten(), 
        os.path.join(plot_dir, "gmm_comparison_1d_pfode.png")
    )
    plot_gmm_comparison(
        x0_real.flatten(), 
        x0_fake_sde.flatten(), 
        os.path.join(plot_dir, "gmm_comparison_1d_sde.png")
    )
elif CONFIG['data_dim'] >= 2:
    # 2D: Plot Real, PFODE, and SDE on one figure
    plot_gmm_comparison_2d_combined(
        x0_real, # Shape (N, D)
        x0_fake_pfode, # Shape (N, D)
        x0_fake_sde, # Shape (N, D)
        os.path.join(plot_dir, "gmm_comparison_2d_combined.png")
    )

# --- Plot 2: Trajectories ---
if CONFIG['data_dim'] == 1:
    print("Plotting 1D trajectories (vs. time)...")
    plot_1d_trajectories(
        pfode_paths[:100], 
        gld.ts.cpu().numpy(),
        data_dim=CONFIG['data_dim'],
        save_path=os.path.join(plot_dir, "trajectories_1d_pfode.png")
    )
    plot_1d_trajectories(
        reverse_sde_paths[:100],
        gld.ts.cpu().numpy(),
        data_dim=CONFIG['data_dim'],
        save_path=os.path.join(plot_dir, "trajectories_1d_sde.png")
    )
elif CONFIG['data_dim'] >= 2:
    print("Plotting 2D trajectories (phase space)...")
    # 2D: Plot PFODE and SDE trajectories side-by-side
    plot_2d_trajectories_combined(
        pfode_paths[:100], # Plot 100 paths
        reverse_sde_paths[:100],
        save_path=os.path.join(plot_dir, "trajectories_2d_combined.png")
    )

# --- Plot 3: Animations ---
if CONFIG['data_dim'] >= 2:
    print("Creating combined animation... (This may take a while)")
    create_animation_combined(
        pfode_paths,
        reverse_sde_paths,
        gld.ts,
        CONFIG['data_dim'],
        save_path=os.path.join(plot_dir, "animation_combined.gif")
    )

print(f"✅ Evaluation complete. All plots and animations saved in: {plot_dir}")