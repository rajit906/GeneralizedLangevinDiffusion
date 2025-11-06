# datasets/gmm1d.py
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class GMM1DDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for a 1D Gaussian Mixture Model.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.train_data = None
        self.val_data = None

    def setup(self, stage=None):
        """
        Generate the 1D GMM dataset and split into train/validation sets.
        """
        n_points = self.cfg.data.n_points
        means = np.array(self.cfg.data.means)  # list of means
        stds = np.array(self.cfg.data.stds)    # list of std deviations
        weights = np.array(self.cfg.data.weights)  # list of mixture weights
        assert len(means) == len(stds) == len(weights), "Means, stds, and weights must have same length"

        # Sample from the mixture
        components = np.random.choice(len(means), size=n_points, p=weights)
        X = np.random.randn(n_points) * stds[components] + means[components]
        X = X.reshape(-1, 1)  # Make it 2D for consistency (n_samples, 1)
        data = torch.from_numpy(X).float()

        # Train/validation split (90%/10%)
        n_train = int(len(data) * 0.9)
        self.train_data, self.val_data = torch.utils.data.random_split(data, [n_train, len(data) - n_train])

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            num_workers=0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=1,
            shuffle=False
        )


# --- Main test script ---
if __name__ == "__main__":
    from omegaconf import OmegaConf

    mock_cfg = OmegaConf.create({
        "data": {
            "n_points": 500,
            "means": [-10., 10.],
            "stds": [0.5, 0.5],
            "weights": [0.5, 0.5]
        },
        "training": {
            "batch_size": 128
        }
    })

    datamodule = GMM1DDataModule(mock_cfg)
    datamodule.setup()

    print("--- Training Data ---")
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print(batch)  # batch is already a tensor of shape (batch_size, 1)
        break

    print("\n--- Validation Data ---")
    val_loader = datamodule.val_dataloader()
    for batch in val_loader:
        print(batch)
        break
