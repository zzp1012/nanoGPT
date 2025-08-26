"""
Script to load a checkpoint and compute gradients on random batch.
This is useful for gradient analysis, debugging, or other gradient-based computations.

Usage:
$ python compute_gradients.py --checkpoint_path=out/ckpt.pt --batch_size=32
"""

import os
import argparse
import numpy as np
import pickle
import random
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm

from utils.model_llama import build_llama_model

def compute_noise_scale(batch_updates, global_updates, batch_size):
    """
    Compute noise scale for each parameter using efficient covariance computation.
    Noise scale = trace(covariance_matrix) / squared norm of true update
    
    Args:
        batch_updates: List of updates for each batch
        global_updates: Dictionary of global updates (true updates)
    
    Returns:
        Dictionary mapping parameter names to their noise scales
    """
    noise_scales = {}
    overall_true_gradient_norm_squared = 0
    overall_covariance_matrix_trace = 0
    for name, updates in batch_updates.items():
        if name not in global_updates:
            continue

        num_batches = len(updates)
        updates_2d = torch.stack(updates).view(num_batches, -1)

        # equation: noise_scale
        true_gradient_norm_squared = 1/(num_batches * batch_size - batch_size) *(
            num_batches * batch_size * torch.sum(global_updates[name] ** 2) - batch_size * torch.mean(torch.sum(updates_2d ** 2, dim=1))
        )
        covariance_matrix_trace = 1/(1/batch_size - 1/(num_batches * batch_size)) * (
            torch.mean(torch.sum(updates_2d ** 2, dim=1)) - torch.sum(global_updates[name] ** 2)
        )
        noise_scale = covariance_matrix_trace / true_gradient_norm_squared
        noise_scales[name] = noise_scale.item()

        # update overall true gradient norm squared and covariance matrix trace
        overall_true_gradient_norm_squared += true_gradient_norm_squared
        overall_covariance_matrix_trace += covariance_matrix_trace

    # compute overall noise scale
    overall_noise_scale = overall_covariance_matrix_trace / overall_true_gradient_norm_squared
    
    return noise_scales, overall_noise_scale


def get_batch(data: np.ndarray, 
              batch_size: int, 
              block_size: int, 
              device: str, 
              device_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a random batch from the data."""
    # Create random indices for batch sampling
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Create input and target sequences
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    # Move to device
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def set_seed(seed):
    """
    Set the seed for the random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Load checkpoint and compute gradients on random batch'
    )
    parser.add_argument('--checkpoint_path', '-d', type=str, required=True, 
                       help='Path to the checkpoint file')
    parser.add_argument('--output_path', type=str, default=None, 
                       help='Path to save the output file')
    parser.add_argument('--batch_size', type=int, default=12, 
                       help='Batch size for gradient computation')
    parser.add_argument('--accumulation_steps', type=int, default=50,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    noise_scales = {}
    overall_noise_scales = {}
    for ckpt_name in os.listdir(args.checkpoint_path):
        ckpt_path = os.path.join(args.checkpoint_path, ckpt_name)
        iter_name = int(ckpt_name.split('_')[-1].split('.')[0])
        print(f"Processing checkpoint at iteration {iter_name}...")
        
        # load checkpoint
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            config = checkpoint['config']
        except Exception as e:
            print(f"Error loading checkpoint at {ckpt_path}: {e}")
            continue

        # get the detailed config
        dataset = config.get('dataset', 'finewebedu')
        model_name = config.get('model_name', '170M')
        device = config.get('device', 'cuda')
        dtype = config.get('dtype', 'bfloat16')
        block_size = config.get('block_size', 1024)

        # device setup
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # load data
        print(f"Loading data from {dataset}...")
        data_dir = os.path.join('data', dataset)
        trainset = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

        # load model
        print(f"Loading model {model_name}...")
        model = build_llama_model(model_name)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        
        # set model to training mode
        model.train()

        # sample multiple batches
        gradient_dict = {}
        
        for ii in tqdm(range(args.accumulation_steps), desc="Processing batches"):
            # fetch data
            X, Y = get_batch(
                trainset, args.batch_size, block_size, device, device_type
            )
        
            # zero gradients manually
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()

            # forward pass
            with ctx:
                logits = model(X).logits.view(args.batch_size * block_size, -1)
            loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1)

            # backward pass
            loss.backward()

            # save gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if name not in gradient_dict:
                        gradient_dict[name] = []
                    gradient_dict[name].append(param.grad.clone().cpu())

        # compute global gradients
        global_grad_dict = {}
        for name, grads in gradient_dict.items():
            global_grad_dict[name] = torch.stack(grads).mean(dim=0)

        # compute noise scales
        print("Computing noise scales...")
        noise_scales[iter_name], overall_noise_scales[iter_name] = compute_noise_scale(gradient_dict, global_grad_dict, args.batch_size)

    # save noise scales
    with open(args.output_path, 'wb') as f:
        pickle.dump((noise_scales, overall_noise_scales), f)


if __name__ == '__main__':
    main()
