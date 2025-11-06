# utils/data.py
import torch
from torch.utils.data import Dataset

class GMMDataset(Dataset):
    """
    A torch.utils.data.Dataset to sample from a 1D or 2D 
    Gaussian Mixture Model.
    """
    def __init__(self, gmm_params, n_samples):
        super().__init__()
        self.n_samples = n_samples
        self.weights = torch.tensor(gmm_params['weights'], dtype=torch.float32)
        self.means = torch.tensor(gmm_params['means'], dtype=torch.float32)
        self.stds = torch.tensor(gmm_params['stds'], dtype=torch.float32)
        
        # Pre-sample all data
        self.samples = self._get_initial_samples(n_samples)

    def _get_initial_samples(self, n_samples):
        """Samples from the GMM distribution."""
        counts = torch.multinomial(self.weights, n_samples, replacement=True)
        
        # Determine data dimension from the means tensor
        if self.means.dim() == 1:
            # 1D case (e.g., means = [-10., 10.])
            data_dim = 1
        else:
            # ND case (e.g., means = [[-5., -5.], ...])
            data_dim = self.means.shape[1]

        samples_list = []
        for i in range(len(self.weights)):
            n_i = int((counts == i).sum())
            if n_i == 0:
                continue
            
            # Sample N_i points from component i
            if data_dim == 1:
                noise = torch.randn(n_i)
            else:
                noise = torch.randn(n_i, data_dim) # (N_i, D)
            
            # Scale by std and shift by mean
            # self.stds[i] is a scalar, self.means[i] is (D,)
            sample_i = noise * self.stds[i] + self.means[i]
            samples_list.append(sample_i)

        samples = torch.cat(samples_list, dim=0)
        
        # Ensure 1D data is also (N, 1)
        if data_dim == 1:
            samples = samples.unsqueeze(-1)
            
        # Shuffle
        samples = samples[torch.randperm(n_samples)]
        return samples # Shape (n_samples, data_dim)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_all_samples(self):
        return self.samples