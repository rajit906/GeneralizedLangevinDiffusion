import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

from diffusion.gld_image import GeneralizedLangevinDiffusion
from models import GLDScoreUNet # Import the new UNet
from utils.viz import plot_losses # We only need loss plots for now

# Use cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- 1. Configuration ---
CONFIG = {
    # Data
    'data_dim': 1,  # <-- 1 for MNIST (channel dim)
    'img_shape': (1, 28, 28),
    'data_path': './data',
    # Training
    'do_train': True,
    'n_epochs': 25, # Start with 25
    'batch_size': 128,
    'lr': 1e-3,
    # Diffusion
    'n_steps': 500,
    'T': 1.0,
    'gamma': 1.0,
    'lambda_val': 1.0,
    'c_val': 0.1,
    'M': 1.0,
    # Evaluation
    'n_eval_samples': 64, # Just enough for a grid
}

# --- 2. Setup Directories ---
subfolder_name = (
    f"mnist_dim{CONFIG['data_dim']}_gamma{CONFIG['gamma']}_lambda{CONFIG['lambda_val']}_"
    f"c{CONFIG['c_val']}_M{CONFIG['M']}_lr{CONFIG['lr']}_bs{CONFIG['batch_size']}"
)
save_dir = f"checkpoints/gld_mnist/{subfolder_name}"
plot_dir = os.path.join(save_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)
checkpoint_path = os.path.join(save_dir, "score_net_final.pt")

# --- 3. Load Data ---
print("Loading MNIST data...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(
    root=CONFIG['data_path'], 
    train=True, 
    download=True, 
    transform=transform
)
val_dataset = datasets.MNIST(
    root=CONFIG['data_path'], 
    train=False, 
    download=True, 
    transform=transform
)

# Use a subset for faster validation
val_subset = torch.utils.data.Subset(val_dataset, range(2000))

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=True, 
    num_workers=4,
    pin_memory=True
)
val_loader = DataLoader(
    val_subset, 
    batch_size=CONFIG['batch_size'], 
    shuffle=False,
    num_workers=4,
    pin_memory=True
)
print(f"Data loaded: {len(train_dataset)} train, {len(val_subset)} val samples.")


# --- 4. Initialize Model & Precompute ---
print("Initializing model...")
gld = GeneralizedLangevinDiffusion(
    data_dim=CONFIG['data_dim'], # Channel dim
    n_steps=CONFIG['n_steps'],
    T=CONFIG['T'],
    gamma=CONFIG['gamma'],
    lambda_val=CONFIG['lambda_val'],
    c_val=CONFIG['c_val'],
    M=CONFIG['M'],
    device=DEVICE
)

gld.precompute()

# --- 5. Training ---
if CONFIG['do_train']:
    print("Starting training...")
    
    # Pass UNet arguments
    model_kwargs = {
        'in_channels': 2 * CONFIG['data_dim'], # 2*C for (p, s)
        'out_channels': 2 * CONFIG['data_dim'],
        'base_dim': 32
    }
    
    model, (train_losses, val_losses) = gld.train_score_network(
        ScoreNetwork=GLDScoreUNet,
        dataloader=train_loader,
        val_dataloader=val_loader,
        n_epochs=CONFIG['n_epochs'],
        lr=CONFIG['lr'],
        **model_kwargs # Pass args to UNet constructor
    )

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
model_kwargs = {
    'in_channels': 2 * CONFIG['data_dim'],
    'out_channels': 2 * CONFIG['data_dim'],
    'base_dim': 32
}
model = GLDScoreUNet(**model_kwargs).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.eval()
print("✅ Checkpoint loaded successfully.")

# --- 7. Evaluation & Plotting ---
print(f"Generating {CONFIG['n_eval_samples']} samples...")
with torch.no_grad():
    pfode_paths = gld.generate_samples(
        CONFIG['n_eval_samples'], 
        CONFIG['img_shape'],
        model, 
        method='pfode'
    ) # (B, n_steps, 3C, H, W)

# Extract generated x0 samples
x0_fake = pfode_paths[:, 0, :CONFIG['data_dim'], :, :] # (B, C, H, W)
x0_fake = x0_fake.cpu().numpy()

# Denormalize from [-1, 1] to [0, 1] for plotting
x0_fake = (x0_fake * 0.5) + 0.5
x0_fake = np.clip(x0_fake, 0, 1)

# Plot a grid of samples
n_rows = int(np.sqrt(CONFIG['n_eval_samples']))
fig, axs = plt.subplots(n_rows, n_rows, figsize=(10, 10))
for i in range(n_rows):
    for j in range(n_rows):
        idx = i * n_rows + j
        axs[i, j].imshow(x0_fake[idx, 0], cmap='gray')
        axs[i, j].axis('off')
plt.suptitle('Generated MNIST Samples (PFODE)')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "mnist_samples.png"))
plt.close()

print(f"✅ Evaluation complete. Sample grid saved in: {plot_dir}")