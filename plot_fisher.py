import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# define Gloabl variables
FISHER_MAT_DIR = "/jfs-dialogue-mmos02-rs02/users/Zp/nanoGPT/out/llama_ratio/0.25B/baseline/fisher_mat"
SAVE_PATH = "/jfs-dialogue-mmos02-rs02/users/Zp/nanoGPT/out/fisher_mat_plot/llama/0.25B/baseline/"
ITERATIONS = [2000]

def preprocess_fisher_mat(fisher_mat: dict):
    """
    Preprocess the fisher matrix to make it a square matrix.
    """
    if "llama" in FISHER_MAT_DIR:
        name_dict = {
            'Embed': ['embed_tokens.weight'],
            'QK': ['self_attn.q_proj.weight', 'self_attn.k_proj.weight'], 
            'FFN': ['mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight'],
            'Head': ['lm_head.weight'], 
            'VO': ['self_attn.v_proj.weight', 'self_attn.o_proj.weight'], 
            'Norm': ['input_layernorm.weight', 'post_attention_layernorm.weight']
        }
    elif "minimax" in FISHER_MAT_DIR:
        name_dict = {
            "Embed": ['embed_tokens.weight'],
            "QK": ['self_attn.q_proj.weight', 'self_attn.k_proj.weight'],
            "MLP": ['w1.weight', 'w2.weight', 'w3.weight'],
            "VO": ['self_attn.v_proj.weight', 'self_attn.o_proj.weight'],
            # "Router": ['block_sparse_moe.gate.weight'], 
            "LayerNorm": ['input_layernorm.weight', 'post_attention_layernorm.weight'],
            "RMSNorm": ['model.norm.weight'],
            "Head": ['lm_head.weight'],
        }
    else:
        raise ValueError(f"Unknown model: {FISHER_MAT_DIR}")
    
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
        size = fisher.size
        print(f"{block_type}: {size}")
        selected_index = np.random.choice(size, size=num_selected, replace=size <= num_selected)
        selected_fisher_mat[block_type] = np.maximum(fisher[selected_index], eps)
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
