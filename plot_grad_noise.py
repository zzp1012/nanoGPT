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
}

def plot_overall_grad_noise(overall_grad_noise_dict: dict[str, dict[str, float]], 
                            save_path: Path):
    """
    Plot the overall gradient noise for each layer.
    Args:
        overall_grad_noise_dict: A dictionary of gradient noise for each layer.
        save_path: The path to save the plot.
    """
    plt.figure(figsize=(8, 5))
    for key_name, grad_noise_dict in overall_grad_noise_dict.items():
        plt.plot(grad_noise_dict.keys(), grad_noise_dict.values(), label=key_name)
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
    filename = "gradient_noise.pkl"
    for dirname in os.listdir(args.data_dir):
        path = data_dir / dirname / filename
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

        # get the global batch size from the dirname
        try:
            gbs = dirname.split("_")[2]
            global_batch_size = GLOBAL_BATCH_SIZE[gbs]
        except KeyError:
            print(f"Unknown global batch size: {dirname.split('_')[2]}")
            continue

        optimizer_name = dirname.split("_")[1]
        key_name = f"{optimizer_name}_{gbs}"

        overall_grad_noise_dict[key_name] = dict(sorted({
            iter_name*global_batch_size: grad_noise for iter_name, grad_noise in data[1].items()
        }.items(), key=lambda x: x[0]))

    plot_overall_grad_noise(overall_grad_noise_dict, save_dir / "overall_grad_noise.png")


if __name__ == "__main__":
    main()