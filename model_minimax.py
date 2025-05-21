from transformers import AutoModelForCausalLM, AutoConfig

def build_minimax_model(model_name: str):
    """
    Build a MiniMax model from a given model name.
    """
    config = AutoConfig.from_pretrained("MiniMaxAI/MiniMax-Text-01", trust_remote_code=True)

    if model_name == "0.25B_dense":
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.max_position_embeddings = 1024
        config.vocab_size = 50304
        config.num_local_experts = 1
        config.num_experts_per_tok = 1
        config.attn_type_list = [1] * 12
    elif model_name == "0.25B_moe":
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.intermediate_size = 3072
        config.num_attention_heads = 12
        config.num_key_value_heads = 12
        config.max_position_embeddings = 1024
        config.vocab_size = 50304
        config.num_local_experts = 4
        config.num_experts_per_tok = 1
        config.attn_type_list = [1] * 12
    else:
        raise ValueError(f"Model name {model_name} not supported")
    
    # print the total number of parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters())
    
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    print(f"Total number of parameters: {count_parameters(model)}")

    return model
