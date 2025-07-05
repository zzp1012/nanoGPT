from transformers import AutoConfig
from modeling_minimax_text_01 import MiniMaxText01ForCausalLM

def build_minimax_model(model_name: str, block_size: int = 1024, vocab_size: int = 50304):
    """
    Build a MiniMax model from a given model name.
    """
    config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-Text-01", trust_remote_code=True)

    if model_name == "0.25B_dense":
        config.num_hidden_layers = 8
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.max_position_embeddings = block_size
        config.vocab_size = vocab_size
        config.num_local_experts = 1
        config.num_experts_per_tok = 1
        config.attn_type_list = [1] * 8
        config.router_aux_loss_coef = 0.0
        config.postnorm = False
        config.initializer_range = 0.06
    elif model_name == "0.25B_moe":
        config.num_hidden_layers = 8
        config.hidden_size = 1024
        config.intermediate_size = 4096
        config.num_attention_heads = 8
        config.num_key_value_heads = 8
        config.max_position_embeddings = block_size
        config.vocab_size = vocab_size
        config.num_local_experts = 8
        config.num_experts_per_tok = 2
        config.attn_type_list = [1] * 8
        config.postnorm = False
        config.initializer_range = 0.06
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    # print the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    model = MiniMaxText01ForCausalLM(config)
    print(f"Total number of parameters: {count_parameters(model)}")

    return model
