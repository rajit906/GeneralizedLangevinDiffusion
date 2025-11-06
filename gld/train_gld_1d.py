# train_gld.py
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt

from sde_lib import GLDSDE
from models import get_model
from utils.util import EMA, calculate_mmd, CustomProgressBar
from datasets.gmm1d import GMM1DDataModule


class GLDSDEModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        self.data_dim = cfg.model.data_dim
        self.ps_dim = cfg.model.ps_dim
        
        self.model = get_model(cfg.model.name, config=cfg.model)
        self.ema = EMA(self.model, decay=cfg.training.ema_decay)
        self.sde = None

    def setup(self, stage=None):
        self.sde = GLDSDE(
            data_dim=self.data_dim,
            device=self.device,
            **self.cfg.sde
        )

    def forward(self, ps, t):
        return self.model(ps, t)

    def training_step(self, batch, _batch_idx):
        # We DON'T use the batch. We sample directly from the marginals.
        B = self.cfg.training.batch_size
        
        # 1. Sample time
        t_idx = torch.randint(0, self.sde.N, (1,), device=self.device).item()
        t_scalar = self.sde.ts[t_idx].item()
        t_batch = torch.full((B,), t_scalar, device=self.device)

        # 2. Get GMM marginal parameters for time t
        # Pass the 'data' config sub-dict
        weights_K, means_K, covs_K = self.sde.get_perturbed_gmm_params(
            t_scalar, self.cfg.data
        ) # (K,), (K, 3), (K, 3, 3) for 1D case

        # 3. Sample mixture components for the batch
        comp_idx = torch.multinomial(weights_K, B, replacement=True) # (B,)
        mu_t_batch = means_K[comp_idx]   # (B, 3)
        Sigma_t_batch = covs_K[comp_idx] # (B, 3, 3)

        # 4. Cholesky and Reparameterization
        # Jitter was already added in get_perturbed_gmm_params
        L_t_batch = torch.linalg.cholesky(Sigma_t_batch) # (B, 3, 3)
        L_t_trans = L_t_batch.transpose(1, 2)            # (B, 3, 3)
        
        eps = torch.randn(B, self.sde.state_dim, device=self.device) # (B, 3)
        
        # z_t = mu_t_batch + eps @ L_t_trans
        z_t = mu_t_batch + torch.bmm(eps.unsqueeze(1), L_t_trans).squeeze(1) # (B, 3)

        # --- Loss Calculation (from original code) ---
        
        # 5. Target score
        rhs = -eps.unsqueeze(-1) # (B, 3, 1)
        score_full_target = torch.linalg.solve(L_t_trans, rhs).squeeze(-1) # (B, 3)
        # target_ps = score_full_target[:, self.data_dim:] # (B, 2) # Not needed

        # 6. Prediction
        ps_inputs = z_t[:, self.data_dim:] # (B, 2)
        score_ps_pred = self.model(ps_inputs, t_batch) # (B, 2)

        # 7. Reparameterization to eps
        score_full_pred = torch.cat(
            [torch.zeros(B, self.data_dim, device=self.device), score_ps_pred],
            dim=1
        ) # (B, 3)
        
        # eps_hat = - L_t_trans @ score_full_pred
        eps_hat = -torch.bmm(L_t_trans, score_full_pred.unsqueeze(-1)).squeeze(-1) # (B, 3)

        eps_target_ps = eps[:, self.data_dim:] # (B, 2)
        eps_hat_ps   = eps_hat[:, self.data_dim:] # (B, 2)

        # 8. Weighted loss
        diff = eps_hat_ps - eps_target_ps # (B, 2)
        GGt_ps = self.sde.GGt_ps          # (2, 2)
        
        per_sample_loss = torch.einsum("bi,ij,bj->b", diff, GGt_ps, diff) # (B,)
        
        # 9. [FIX] ADD BACK THE SCALING FACTOR
        L_ps = L_t_batch[:, self.data_dim:, self.data_dim:] # (B, 2, 2)
        L_norm_sq = torch.einsum("bij,kl,bkl->b", L_ps, GGt_ps, L_ps) # (B,)
        
        # Add a small epsilon to prevent division by zero
        per_sample_loss = per_sample_loss / (L_norm_sq + 1e-8)
        loss = per_sample_loss.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_before_optimizer_step(self, _optimizer):
        self.ema.update()

    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, _batch_idx):
        self.validation_step_outputs.append(batch)

    def on_validation_epoch_end(self):
        outputs = torch.cat(self.validation_step_outputs, dim=0)
        B = outputs.shape[0]

        self.ema.apply_shadow()
        fake = self.sde.ode_sampler(self.model, (B, self.data_dim), steps=self.cfg.training.n_val_steps)
        self.ema.restore()

        mmd_value = calculate_mmd(outputs.cpu(), fake.cpu())
        self.log("val_mmd", mmd_value, prog_bar=True)

        real_np = outputs.cpu().numpy().flatten()
        fake_np = fake.cpu().numpy().flatten()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(real_np, bins=60, density=True, alpha=0.6, label="Real")
        ax.hist(fake_np, bins=60, density=True, alpha=0.6, label="Generated (GLD)")
        ax.set_title(f"Epoch {self.current_epoch+1} | MMD={mmd_value:.4f}")
        ax.legend()
        self.logger.experiment.log({"samples_hist": wandb.Image(fig)})
        plt.close(fig)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.training.lr)


@hydra.main(config_path="configs", config_name="gldsde_gmm1d")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(42)
    datamodule = GMM1DDataModule(cfg)
    model = GLDSDEModel(cfg)

    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.wandb.name}",
        filename="{epoch}-{val_mmd:.4f}",
        monitor="val_mmd",
        mode="min",
        save_top_k=3
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, CustomProgressBar(refresh_rate=10)],
        check_val_every_n_epoch=cfg.training.val_every_n_epoch,
        limit_train_batches=cfg.training.n_steps_per_epoch
    )

    trainer.fit(model, datamodule)
    wandb.finish()


if __name__ == "__main__":
    main()
