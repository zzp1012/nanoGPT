from transformers import AutoConfig
from utils.model_minimax.modeling_minimax_text_01 import MiniMaxText01ForCausalLM

def build_minimax_model(model_name: str, use_combined_qkv: bool = False, use_combined_w1w3: bool = False, use_QK_norm: bool = True):
    """
    Build a MiniMax model from a given model name.
    """
    config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-Text-01", trust_remote_code=True)
    vocab_size = 50304
    block_size = 1024
    if model_name == "170M":
        config.num_hidden_layers = 8
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.head_dim = 128
        config.max_position_embeddings = block_size
        config.vocab_size = vocab_size
        config.num_local_experts = 1
        config.num_experts_per_tok = 1
        config.attn_type_list = [1] * 8
        config.router_aux_loss_coef = 0.0
        config.postnorm = False
        config.initializer_range = 0.06
        config.use_combined_qkv = use_combined_qkv
        config.use_combined_w1w3 = use_combined_w1w3
        config.use_QK_norm = use_QK_norm
    else:
        raise ValueError(f"Model name {model_name} not supported")

    print(f"FFN/hidden_size: {config.intermediate_size / config.hidden_size}")
    print(f"FFN/num_heads/head_dim: {config.intermediate_size / config.num_attention_heads / config.head_dim}")
    print(f"head_dim * num_heads / hidden_size: {config.head_dim * config.num_attention_heads / config.hidden_size}")
    
    model = MiniMaxText01ForCausalLM(config)
    # print the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {count_parameters(model)}")
    return model
