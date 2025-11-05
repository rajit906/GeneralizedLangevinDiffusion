# train.py
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import matplotlib.pyplot as plt

from sde_lib import VPSDE
from models import get_model
from utils.util import EMA, calculate_mmd, CustomProgressBar
from datasets.swissroll import SwissRollDataModule

class SDEModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        
        # Initialize model and EMA. SDE will be initialized in setup().
        self.model = get_model(cfg.model.name, config=cfg.model)
        self.ema = EMA(self.model, decay=cfg.training.ema_decay)
        self.sde = None

    def setup(self, stage=None):
        # This hook is called after the model is moved to the correct device.
        # It's the ideal place to initialize objects that need the device context.
        self.sde = VPSDE(
            beta_min=self.cfg.sde.beta_min,
            beta_max=self.cfg.sde.beta_max,
            N=self.cfg.sde.N,
            device=self.device
        )

    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, _batch_idx):
        x0 = batch
        t = torch.rand(x0.shape[0], device=self.device) * (1. - 1e-5) + 1e-5
        mean, std = self.sde.marginal_prob(x0, t)
        z = torch.randn_like(x0)
        xt = mean + std * z
        predicted_z = self(xt, t)
        _, g = self.sde.sde(torch.zeros_like(xt), t)
        loss_weight = (g**2) / (std.squeeze()**2 + 1e-8)
        squared_error = (predicted_z - z) ** 2
        loss = torch.mean(loss_weight[:, None] * squared_error)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_before_optimizer_step(self, _optimizer):
        # Update EMA weights before the optimizer step
        self.ema.update()

    def on_validation_epoch_start(self):
        """
        Called at the start of the validation epoch.
        """
        # Create an empty list to store the outputs of each validation step.
        self.validation_step_outputs = []

    def validation_step(self, batch, _batch_idx):
        """
        Called for each batch in the validation set.
        """
        self.validation_step_outputs.append(batch)

    def on_validation_epoch_end(self):
        """
        Called once at the end of the validation loop.
        'outputs' is a list of all the batches returned from validation_step.
        """
        outputs = self.validation_step_outputs
        
        if not outputs:
            return

        # Step 1: Aggregate all real samples that were collected.
        real_samples = torch.cat(outputs, dim=0)
        num_real_samples = real_samples.shape[0]

        # Step 2: Generate all fake samples in a single batch operation.
        print(f"Generating {num_real_samples} samples in a single batch...")
        self.ema.apply_shadow()
        shape = (num_real_samples, self.cfg.model.input_dim)
        fake_samples = self.sde.reverse_sde_sampler(self.model, shape, steps=500)
        self.ema.restore()
        
        # Step 3: Calculate metrics and create plots as before.
        mmd_value = calculate_mmd(real_samples, fake_samples)
        self.log('val_mmd', mmd_value, on_epoch=True, prog_bar=True, logger=True)
        
        viz_samples = fake_samples[:self.cfg.training.n_samples].cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(viz_samples[:, 0], viz_samples[:, 1], alpha=0.6, s=15, color='blue', label='Generated')
        ax.scatter(real_samples[:self.cfg.training.n_samples, 0].cpu(), real_samples[:self.cfg.training.n_samples, 1].cpu(), alpha=0.6, s=15, color='green', label='Real')
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
    

@hydra.main(config_path="configs", config_name="vpsde_swiss_roll")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    
    # Setup reproducibility and data modules
    pl.seed_everything(42)
    datamodule = SwissRollDataModule(cfg)
    model = SDEModel(cfg)

    # Setup logger and callbacks
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="{epoch}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss"
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.n_epochs,
        accelerator="auto",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, CustomProgressBar(refresh_rate=10)],
        log_every_n_steps=10,
        check_val_every_n_epoch=1
        #num_sanity_val_steps=-1
    )

    # Start training
    trainer.fit(model, datamodule)

    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()