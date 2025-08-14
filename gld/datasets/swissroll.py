import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_swiss_roll

class SwissRollDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data = None

    def setup(self, stage=None):
        X, _ = make_swiss_roll(n_samples=self.cfg.data.n_points, noise=self.cfg.data.noise)
        # Normalize to [-1, 1] range
        X = X[:, [0, 2]] / 7.5
        data = torch.from_numpy(X).float()
        self.train_data, self.val_data = torch.utils.data.random_split(
                                        data, [int(len(data)*0.9), len(data) - int(len(data)*0.9)])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.cfg.training.batch_size, shuffle=True, num_workers=4)
        
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=1)
    

# --- Main test script ---
if __name__ == "__main__":
    # Create a mock config for testing purposes
    mock_config = OmegaConf.create({
        "data": {
            "n_points": 100,  # Use a small number of points for a quick test
            "noise": 0.5
        },
        "training": {
            "batch_size": 32
        }
    })

    print("--- Initializing DataModule ---")
    datamodule = SwissRollDataModule(mock_config)
    
    print("--- Setting up data (creating train/val split) ---")
    datamodule.setup()

    print("\n--- Inspecting Validation Dataloader ---")
    val_loader = datamodule.val_dataloader()

    # Get an iterator for the dataloader
    val_iterator = iter(val_loader)
    
    # Inspect the first 3 batches
    for i in range(3):
        try:
            # The dataloader yields a list where the first element is our tensor
            batch = next(val_iterator)
            data_tensor = batch[0]
            
            print(f"\nBatch {i+1}:")
            print(f"  - Full batch object from DataLoader: {batch}")
            print(f"  - Extracted data tensor: {data_tensor}")
            print(f"  - Shape of data tensor: {data_tensor.shape}")
            
            if data_tensor.ndim != 2:
                print("  - ⚠️  WARNING: Tensor is not 2D. This will cause issues.")
            else:
                print("  - ✅  SUCCESS: Tensor is 2D as expected.")

        except StopIteration:
            print("\nReached end of dataloader.")
            break