from transformers import LlamaConfig, LlamaForCausalLM

def build_llama_model(model_name: str):
    """
    Build a llama model from a given model name.
    """
    d_input = 50304
    max_seq_length = 1024
    extra_config = {}
    if model_name == "93M":
        d_model = 512
        num_heads = 16
        num_layers = 8
        d_ff = d_model * 4
        dropout = 0.0
        head_dim = 64
    elif model_name == "170M":
        d_model = 768
        num_heads = 12
        num_layers = 8
        d_ff = d_model * 4
        dropout = 0.0
        head_dim = 128
    elif model_name == "0.25B":
        d_model = 1024
        num_heads = 16
        num_layers = 8
        d_ff = d_model * 4
        dropout = 0.0
        head_dim = 128
    else:
        raise ValueError(f"Model name {model_name} not supported")

    print(f"FFN/hidden_size: {d_ff / d_model}")
    print(f"FFN/num_heads/head_dim: {d_ff / num_heads / head_dim}")
    print(f"head_dim * num_heads / hidden_size: {head_dim * num_heads / d_model}")
    
    model_args = LlamaConfig(
        head_dim=head_dim,
        vocab_size=d_input, 
        hidden_size=d_model, 
        num_attention_heads=num_heads, 
        attention_dropout=dropout, 
        num_hidden_layers=num_layers, 
        intermediate_size=d_ff, 
        max_position_embeddings=max_seq_length, 
        **extra_config
    )
    print(model_args)
    return LlamaForCausalLM(model_args)