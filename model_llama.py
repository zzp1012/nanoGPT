from transformers import LlamaConfig, LlamaForCausalLM

def build_llama_model(model_name: str):
    """
    Build a llama model from a given model name.
    """
    d_input = 50304
    max_seq_length = 1024
    extra_config = {}
    if model_name == "0.25B":
        d_model = 1024  ## fixed due to d_{kv}
        num_heads = 16  ## fixed due to d_{kv}
        num_layers = 8
        d_ff = d_model * 4
        dropout = 0.0
    elif model_name == "0.5B":
        d_model = 1280  ## fixed due to d_{kv}
        num_heads = 20  ## fixed due to d_{kv}
        num_layers = 15
        d_ff = d_model * 4
        dropout = 0.0
    elif model_name == "0.75B":
        d_model = 1664  ## fixed due to d_{kv}
        num_heads = 26  ## fixed due to d_{kv}
        num_layers = 13
        d_ff = d_model * 4
        dropout = 0.0
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    model_args = LlamaConfig(
        vocab_size=d_input, 
        hidden_size=d_model, 
        num_attention_heads=num_heads, 
        attention_dropout=dropout, 
        num_hidden_layers=num_layers, 
        intermediate_size=d_ff, 
        max_position_embeddings=max_seq_length, 
        **extra_config
    )
    return LlamaForCausalLM(model_args)