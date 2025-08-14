import os
import json
import gc
import configargparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance

# --- Imports from the CLD-SGM repository ---
import sde_lib
import sampling
from util import datasets, utils
from util.checkpoint import restore_checkpoint
from models import ncsnpp
import models.utils as mutils
from models.ema import ExponentialMovingAverage

def setup_parser():
    """
    Sets up a single, unified parser for all configurations.
    """
    p = configargparse.ArgParser(description="FID evaluation script for CLD-SGM.",
                                 ignore_unknown_config_file_keys=True)
    
    # --- Config files ---
    p.add('-cc', is_config_file=True, default='configs/default_cifar10.txt')
    p.add('-sc', is_config_file=True, default='configs/specific_cifar10.txt')

    # --- User-facing evaluation arguments ---
    p.add('--ckpt_path', type=str, default='checkpoints/cifar10_800k.pth', help='Path to the model checkpoint.')
    p.add('--sampler', type=str, choices=['sscs', 'em', 'ode'], required=True, help='Which sampler to use.')
    p.add('--n_steps', type=int, required=True, help='Number of discrete steps for the sampler.')
    p.add('--batch_size', type=int, default=64, help='Batch size for sampling.')
    p.add('--num_fid_samples', type=int, default=10000, help='Number of samples to generate for FID.')
    p.add('--output_dir', type=str, default='evaluation_results', help='Directory to save results.')

    # --- All original arguments from the notebook ---
    # This is necessary to correctly build the model and other components from the config files.
    p.add('--root', default='.')
    p.add('--workdir', default='work_dir')
    p.add('--mode', choices=['train', 'eval', 'continue'], default='eval')
    p.add('--distributed', action='store_false')
    p.add('--seed', type=int, default=0)
    p.add('--dataset', type=str)
    p.add('--is_image', action='store_true')
    p.add('--image_size', type=int)
    p.add('--center_image', action='store_true')
    p.add('--image_channels', type=int)
    p.add('--data_location', default=None)
    p.add('--sde', type=str)
    p.add('--beta_type', type=str)
    p.add('--beta0', type=float)
    p.add('--beta1', type=float)
    p.add('--m_inv', type=float)
    p.add('--gamma', type=float)
    p.add('--numerical_eps', type=float)
    p.add('--optimizer', type=str)
    p.add('--learning_rate', type=float)
    p.add('--weight_decay', type=float)
    p.add('--grad_clip', type=float)
    p.add('--cld_objective', choices=['dsm', 'hsm'], default='hsm')
    p.add('--loss_eps', type=float)
    p.add('--weighting', choices=['likelihood', 'reweightedv1', 'reweightedv2'])
    p.add('--name', type=str)
    p.add('--ema_rate', type=float)
    p.add('--normalization', type=str)
    p.add('--nonlinearity', type=str)
    p.add('--n_channels', type=int)
    p.add('--ch_mult', type=str)
    p.add('--n_resblocks', type=int)
    p.add('--attn_resolutions', type=str)
    p.add('--resamp_with_conv', action='store_true')
    p.add('--use_fir', action='store_true')
    p.add('--fir_kernel', type=str)
    p.add('--skip_rescale', action='store_true')
    p.add('--resblock_type', type=str)
    p.add('--progressive', type=str)
    p.add('--progressive_input', type=str)
    p.add('--progressive_combine', type=str)
    p.add('--attention_type', type=str)
    p.add('--init_scale', type=float)
    p.add('--fourier_scale', type=int)
    p.add('--conv_size', type=int)
    p.add('--dropout', type=float)
    p.add('--mixed_score', action='store_true')
    p.add('--embedding_type', choices=['fourier', 'positional'])
    p.add('--training_batch_size', type=int)
    p.add('--testing_batch_size', type=int)
    p.add('--sampling_batch_size', type=int)
    p.add('--sampling_method', choices=['ode', 'em', 'sscs'], default='ode')
    p.add('--sampling_rtol', type=float, default=1e-5)
    p.add('--sampling_atol', type=float, default=1e-5)
    p.add('--sscs_num_stab', type=float, default=0.)
    p.add('--denoising', action='store_true')
    p.add('--n_discrete_steps', type=int)
    p.add('--striding', choices=['linear', 'quadratic', 'logarithmic'], default='linear')
    p.add('--sampling_eps', type=float)
    
    return p

def main(config):
    """Main evaluation function."""
    # Apply command-line arguments to the config object for clarity
    config.sampling_batch_size = config.batch_size
    config.n_discrete_steps = config.n_steps
    config.sampling_method = config.sampler

    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utils.set_seeds(config.seed, 0)
    config.output_dir = os.path.join('results', config.output_dir)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"🚀 Starting evaluation on device: {config.device}")
    print(f"Sampler: {config.sampling_method}, Steps: {config.n_discrete_steps}, Batch Size: {config.sampling_batch_size}")

    # --- SDE and Model Setup (copied directly from notebook) ---
    beta_fn = utils.build_beta_fn(config)
    beta_int_fn = utils.build_beta_int_fn(config)
    sde = sde_lib.CLD(config, beta_fn, beta_int_fn)

    score_model = mutils.create_model(config).to(config.device)
    optimizer = utils.get_optimizer(config, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.ema_rate)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

    print(f"🔧 Loading model from checkpoint: {config.ckpt_path}")
    state = restore_checkpoint(config.ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())
    score_model.eval()

    # --- Get Sampler and Scaler ---
    inverse_scaler = utils.get_data_inverse_scaler(config)
    sampling_shape = (config.sampling_batch_size, config.image_channels, config.image_size, config.image_size)
    sampler = sampling.get_sampling_fn(config, sde, sampling_shape, config.sampling_eps)

    # --- Generate and Save a Sample Grid ---
    print("🎨 Generating a 8x8 sample grid...")
    with torch.no_grad():
        samples, _, _ = sampler(score_model)
    
    samples_grid = inverse_scaler(samples).clamp(0.0, 1.0)
    sample_filename = os.path.join(config.output_dir, f'samples_{config.sampling_method}_{config.n_discrete_steps}steps.png')
    save_image(samples_grid, sample_filename, nrow=8)
    print(f"✅ Sample grid saved to {sample_filename}")

    # --- FID Calculation with TorchMetrics ---
    print(f"📈 Calculating FID for {config.num_fid_samples} samples...")
    fid_metric = FrechetInceptionDistance(feature=2048).to(config.device)
    
    config.distributed = False
    config.training_batch_size = config.batch_size
    config.testing_batch_size = config.batch_size
    train_queue, _, _ = datasets.get_loaders(config)

    print("Processing real images for FID...")
    for real_batch, _ in tqdm(train_queue):
        real_batch_uint8 = (real_batch * 255).byte().to(config.device)
        fid_metric.update(real_batch_uint8, real=True)
        
    print("Processing generated images for FID...")
    num_generated = 0
    num_rounds = (config.num_fid_samples - 1) // config.sampling_batch_size + 1
    for _ in tqdm(range(num_rounds)):
        with torch.no_grad():
            x, _, _ = sampler(score_model)
        x_uint8 = (inverse_scaler(x).clamp(0.0, 1.0) * 255).byte().to(config.device)
        fid_metric.update(x_uint8, real=False)
        num_generated += x.shape[0]

    fid_score = fid_metric.compute()
    
    # --- Save and Print Results ---
    print("\n" + "="*40)
    print("🎉 Evaluation Complete!")
    print(f"   FID Score: {fid_score:.4f}")
    print("="*40 + "\n")

    with open(os.path.join(config.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Sampler: {config.sampling_method}\n")
        f.write(f"Steps: {config.n_discrete_steps}\n")
        f.write(f"FID Score: {fid_score:.4f}\n")
    print(f"Results saved to {os.path.join(config.output_dir, 'results.txt')}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = setup_parser()
    config = parser.parse_args()
    main(config)