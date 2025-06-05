import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# define Gloabl variables
FISHER_MAT_DIR = "/minimax-dialogue/users/Zp/nanoGPT/out/minimax_dense_ratio/fisher_mat"
SAVE_PATH = "/minimax-dialogue/users/Zp/nanoGPT/out/dense/fisher_mat_plot"
ITERATIONS = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

def preprocess_fisher_mat(fisher_mat: dict):
    """
    Preprocess the fisher matrix to make it a square matrix.
    """
    name_dict = {
        "Embed": ['embed_tokens.weight'],
        "QK": ['self_attn.q_proj.weight', 'self_attn.k_proj.weight'],
        "MLP": ['w1.weight', 'w2.weight', 'w3.weight'],
        "VO": ['self_attn.v_proj.weight', 'self_attn.o_proj.weight'],
        # "Router": ['block_sparse_moe.gate.weight'], 
        "LayerNorm": ['input_layernorm.weight', 'post_attention_layernorm.weight'],
        "RMSNorm": ['model.norm.weight'],
        "Head": ['lm_head.weight']
    }
    
    # concatenate the fisher matrix for each parameter group
    preprocessed_fisher_mat = {}
    for block_type, param_name_list in name_dict.items():
        param_list = []
        for param_name in param_name_list:
            param_list.append(
                torch.concat([fisher for key, fisher in fisher_mat.items() if param_name in key])
            )
        preprocessed_fisher_mat[block_type] = torch.concat(param_list).view(-1).numpy()
    return preprocessed_fisher_mat


def random_selection(fisher_mat: dict, num_selected: int, eps: float = 1e-14):
    """
    Randomly select num_selected parameters from the fisher matrix.
    """
    selected_fisher_mat = {}
    for block_type, fisher in fisher_mat.items():
        attained_index = np.where(fisher > eps)[0]
        print(f"{block_type}: {attained_index.size}")
        selected_index = np.random.choice(attained_index, size=num_selected, replace=attained_index.size < num_selected)
        selected_fisher_mat[block_type] = fisher[selected_index]
    return selected_fisher_mat


def plot_fisher_mat(fisher_mat: dict, save_path: Path):
    """
    Plot the fisher matrix distribution.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.kdeplot(fisher_mat, bw_adjust=0.25, log_scale=True)
    plt.savefig(save_path / f"fisher_mat_{iteration}.png")


if __name__ == "__main__":
    fisher_mat_dir = Path(FISHER_MAT_DIR)
    save_path = Path(SAVE_PATH)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for iteration in ITERATIONS:
        print(f"Processing iteration {iteration}")
        # load the fisher matrix
        fisher_mat_path = fisher_mat_dir / f"grad_dict_{iteration}.pt"
        fisher_mat = torch.load(fisher_mat_path, map_location="cpu")

        # preprocess the fisher matrix
        preprocessed_fisher_mat = preprocess_fisher_mat(fisher_mat)
        for block_type, fisher in preprocessed_fisher_mat.items():
            print(f"{block_type}: {fisher.shape}, {fisher[:10]}")
        selected_fisher_mat = random_selection(preprocessed_fisher_mat, num_selected=10000)
        
        # plot the fisher matrix distribution
        plot_fisher_mat(selected_fisher_mat, save_path)
