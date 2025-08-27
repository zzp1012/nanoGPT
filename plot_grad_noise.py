"""
Plot the gradient noise for each layer.
"""

import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

GLOBAL_BATCH_SIZE = {
    "0.49m": 5 * 8 * 12 * 1024,
    "0.98m": 10 * 8 * 12 * 1024,
    "1.9m": 20 * 8 * 12 * 1024,
    "3.9m": 40 * 8 * 12 * 1024,
    "7.8m": 80 * 8 * 12 * 1024,
}

def plot_overall_grad_noise(overall_grad_noise_dict: dict[str, dict[str, list[float]]], 
                            save_path: Path):
    """
    Plot the overall gradient noise for each layer.
    Args:
        overall_grad_noise_dict: A dictionary of gradient noise for each layer.
        save_path: The path to save the plot.
    """
    plt.figure(figsize=(8, 5))
    for key_name, grad_noise_dict in overall_grad_noise_dict.items():
        x = np.array(list(grad_noise_dict.keys()))
        y = np.array(list(grad_noise_dict.values())) # (num_iters, num_samples)
        y_mu = np.mean(y, axis=1)
        y_std = np.std(y, axis=1)
        plt.plot(x, y_mu, label=key_name)
        plt.fill_between(x, y_mu - y_std, y_mu + y_std, alpha=0.2)
    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.xlabel("Sample")
    plt.ylabel("Gradient Noise Scale")
    plt.title("Overall Gradient Noise Scale")
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the gradient noise for each layer."
    )
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                        help="The directory containing the gradient noise data.")
    parser.add_argument("--save_dir", "-s", type=str, required=True,
                        help="The directory to save the plots.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # load the data
    overall_grad_noise_dict = {}
    for dirname in os.listdir(args.data_dir):
        try:
            gbs = dirname.split("_")[2]
            global_batch_size = GLOBAL_BATCH_SIZE[gbs]
        except KeyError:
            print(f"Unknown global batch size: {dirname.split('_')[2]}")
            continue
        
        optimizer_name = dirname.split("_")[1]
        key_name = f"{optimizer_name}_{gbs}"

        path = data_dir / dirname
        for filename in os.listdir(path):
            if not filename.startswith("gradient_noise_"):
                continue

            data_path = path / filename
            try:
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

            if key_name not in overall_grad_noise_dict:
                overall_grad_noise_dict[key_name] = dict(sorted({
                    iter_name*global_batch_size: [grad_noise] for iter_name, grad_noise in data[1].items()
                }.items(), key=lambda x: x[0]))
            else:
                for iter_name, grad_noise in data[1].items():
                    overall_grad_noise_dict[key_name][iter_name*global_batch_size].append(grad_noise)

    plot_overall_grad_noise(overall_grad_noise_dict, save_dir / "overall_grad_noise.png")


if __name__ == "__main__":
    main()