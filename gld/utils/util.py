import torch
from copy import deepcopy
from pytorch_lightning.callbacks import TQDMProgressBar
from tqdm.auto import tqdm

class CustomProgressBar(TQDMProgressBar):
    """A custom progress bar that disables the validation progress bar."""
    def init_validation_tqdm(self):
        """Override the validation progress bar to disable it."""
        return tqdm(disable=True)

def calculate_mmd(x, y, sigma=None):
        """
        Calculates the Maximum Mean Discrepancy (MMD) between two sets of samples, x and y.
        Uses a Gaussian (RBF) kernel.
        """
        # Set default sigma based on median pairwise distance
        if sigma is None:
            x_pdist = torch.pdist(x)
            y_pdist = torch.pdist(y)
            sigma = torch.median(torch.cat([x_pdist, y_pdist]))

        # Gaussian RBF kernel
        def rbf_kernel(a, b, sigma):
            dist = torch.cdist(a, b, p=2)
            return torch.exp(- (dist ** 2) / (2 * sigma ** 2))

        k_xx = rbf_kernel(x, x, sigma).mean()
        k_yy = rbf_kernel(y, y, sigma).mean()
        k_xy = rbf_kernel(x, y, sigma).mean()
        
        return k_xx + k_yy - 2 * k_xy

# Helper for Exponential Moving Average
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = deepcopy(self.model.state_dict())

    def update(self):
        model_params = self.model.state_dict()
        for name, param in model_params.items():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.original_params = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow)

    def restore(self):
        self.model.load_state_dict(self.original_params)