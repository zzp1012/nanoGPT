out_dir = 'out/llama_bw/0.25B/bw_10_8_6_4_1_1_wo_compile'
eval_interval = 200
log_interval = 1
save_interval = 2000
eval_iters = 100
eval_only = False

wandb_log = True
wandb_project = 'llama_bw'
wandb_run_name = '0.25B_bw_10_8_6_4_1_1_wo_compile'

dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

model_name = "0.25B"
learning_rate = 8e-4
embed_alpha = 10.0
qk_alpha = 8.0
mlp_alpha = 6.0
vo_alpha = 4.0
head_alpha = 1.0
ln_alpha = 1.0
max_iters = 50000
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

warmup_iters = 1000
lr_decay_iters = 50000

compile = False