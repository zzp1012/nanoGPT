out_dir = 'out/minimax_moe_merge'
eval_interval = 200
save_interval = 1000

wandb_log = True
wandb_project = 'minimax'
wandb_run_name = '0.25B_moe_merge'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

model_name = "0.25B_moe"
learning_rate = 6e-4
max_iters = 50000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

decay_lr = True
warmup_iters = 1000
lr_decay_iters = 50000

merge_prob = 0.2
merge_interval = 100