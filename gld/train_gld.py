# train_gld.py
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt

# Import the new GLDSDE class
from sde_lib import GLDSDE 
from models import get_model
from utils.util import EMA, calculate_mmd, CustomProgressBar
from datasets.swissroll import SwissRollDataModule

class GLDSDEModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        # Data dimension (e.g., 2 for Swiss Roll)
        self.data_dim = cfg.model.data_dim
        # (p,s) dimension (e.g., 4 for 2D data)
        self.ps_dim = cfg.model.ps_dim
        
        # Initialize model and EMA. SDE will be initialized in setup().
        # The model takes (p,s) and t as input
        self.model = get_model(cfg.model.name, config=cfg.model)
        self.ema = EMA(self.model, decay=cfg.training.ema_decay)
        self.sde = None

    def setup(self, stage=None):
        # This hook is called after the model is moved to the correct device.
        self.sde = GLDSDE(
            data_dim=self.data_dim,
            device=self.device,
            **self.cfg.sde  # Pass all SDE params from config
        )

    def forward(self, ps, t):
        # ps is (B, ps_dim), t is (B,)
        return self.model(ps, t)

    def training_step(self, batch, _batch_idx):
        x0 = batch # (B, 2)
        B = x0.shape[0]

        # 1. Sample ONE time index for the whole batch
        # This is required for the GLDSDE.marginal_prob to be efficient
        t_idx = torch.randint(0, self.sde.N, (1,), device=self.device).item()
        t_scalar = self.sde.ts[t_idx].item()
        t_batch = torch.full((B,), t_scalar, device=self.device, dtype=torch.float32)

        # 2. Get z0 = [x0, p0, s0]
        z0 = self.sde.get_z0(x0) # (B, 6)

        # 3. Get perturbation kernel p(z_t | z_0)
        # mu_t is (B, 6), L_t is (6, 6)
        mu_t, L_t = self.sde.marginal_prob(z0, t_scalar)

        # 4. Reparametrize: Sample z_t and target eps
        eps = torch.randn_like(z0) # (B, 6)
        # z_t = mu_t + L_t @ eps
        z_t = mu_t + (L_t @ eps.T).T

        # 5. Model prediction
        ps_inputs = z_t[:, self.data_dim:] # (B, 4)
        score_ps_pred = self.model(ps_inputs, t_batch) # (B, 4)

        # 6. Reparametrize prediction to eps_hat (HSM loss)
        score_full_pred = torch.cat(
            [torch.zeros(B, self.data_dim, device=self.device), score_ps_pred], 
            dim=1
        ) # (B, 6)
        
        # eps_hat = -L_t^T @ score_pred
        L_t_trans = L_t.T # (6, 6)
        eps_hat = - (L_t_trans @ score_full_pred.unsqueeze(-1)).squeeze(-1) # (B, 6)

        # 7. Get target and predicted eps for (p,s)
        eps_target_ps = eps[:, self.data_dim:] # (B, 4)
        eps_hat_ps = eps_hat[:, self.data_dim:] # (B, 4)

        # 8. Compute GGt-weighted loss
        GGt_ps = self.sde.GGt_ps # (4, 4)
        diff = eps_hat_ps - eps_target_ps
        # ||diff||_GGt^2 = diff^T @ GGt @ diff
        per_sample_loss = torch.einsum("bi,ij,bj->b", diff, GGt_ps, diff)

        # 9. Loss scaling (from gld.py)
        # ||L_ps||_GGt^2
        # L_ps = L_t[self.data_dim:, self.data_dim:] # (4, 4)
        # L_norm_sq = torch.einsum("ij,kl,kl->", L_ps, GGt_ps, L_ps) # scalar
        
        # loss = (per_sample_loss / (L_norm_sq + 1e-8)).mean()
        # 9. Use the unweighted GGt-norm loss
        loss = per_sample_loss.mean()

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_before_optimizer_step(self, _optimizer):
        self.ema.update()

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, _batch_idx):
        self.validation_step_outputs.append(batch)

    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        
        if not outputs:
            return

        real_samples = torch.cat(outputs, dim=0)
        num_real_samples = real_samples.shape[0]

        print(f"Generating {num_real_samples} samples with GLD Sampler...")
        self.ema.apply_shadow()
        
        # Shape for 'x' (data)
        shape = (num_real_samples, self.cfg.model.data_dim) 
        
        # Use the SDE's reverse sampler
        fake_samples = self.sde.ode_sampler(
            self.model, shape, steps=self.cfg.training.n_val_steps
        )
        self.ema.restore()
        mmd_value = calculate_mmd(real_samples.cpu(), fake_samples.cpu())
        self.log('val_mmd', mmd_value, on_epoch=True, prog_bar=True, logger=True)
        
        # Plotting
        viz_samples = fake_samples[:self.cfg.training.n_plot_samples].cpu().numpy()
        real_viz_samples = real_samples[:self.cfg.training.n_plot_samples].cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(viz_samples[:, 0], viz_samples[:, 1], alpha=0.6, s=15, color='blue', label='Generated (GLD)')
        ax.scatter(real_viz_samples[:, 0], real_viz_samples[:, 1], alpha=0.6, s=15, color='green', label='Real')
        ax.set_title(f'Epoch {self.current_epoch + 1} | MMD: {mmd_value:.4f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.legend()
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        
        self.logger.experiment.log({
            "generated_samples": [wandb.Image(fig)]
        })
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)
        

@hydra.main(config_path="configs", config_name="gldsde_swiss_roll")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    pl.seed_everything(42)
    datamodule = SwissRollDataModule(cfg)
    model = GLDSDEModel(cfg)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.wandb.name}",
        filename="{epoch}-{val_mmd:.4f}",
        save_top_k=3,
        monitor="val_mmd",
        mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, CustomProgressBar(refresh_rate=10)],
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.training.val_every_n_epoch
    )

    trainer.fit(model, datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()