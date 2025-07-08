"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model_minimax import build_minimax_model

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 200
log_interval = 1
save_interval = 2000
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'minimax_bw'
wandb_run_name = '0.25B_dense_baseline' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
vocab_size = 50304
# model
model_name = "0.25B_dense"
# adamw optimizer
learning_rate = 8e-4 # max learning rate
embed_alpha = 1.0
head_alpha = 1.0
ln_alpha = 1.0
qk_alpha = 1.0
vo_alpha = 1.0
mlp_alpha = 1.0
router_alpha = 1.0
rmsn_alpha = 1.0
max_iters = 50000 # total number of training iterations
weight_decay = 0.1
beta1 = 0.90
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 50000 # should be ~= max_iters per Chinchilla
min_lr = learning_rate / 20 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
seed = 41
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

def set_seed(seed):
    """
    Set the seed for the random number generators.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
trainset = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
valset = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = trainset if split == 'train' else valset
    ix_list = []
    for _ in range(ddp_world_size):
        ix_list.append(torch.randint(len(data) - block_size, (batch_size,)))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix_list[ddp_rank]])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init a new model from scratch
print("Initializing a new model from scratch")
model = build_minimax_model(model_name, block_size=block_size, vocab_size=vocab_size)
model.to(device)

# get router_aux_loss_coef
router_aux_loss_coef = model.router_aux_loss_coef

# optimizer
param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
embed_param = [p for n, p in param_dict.items() if 'embed_tokens.weight' in n]
qk_param = [p for n, p in param_dict.items() if 'q_proj' in n or 'k_proj' in n]
vo_param = [p for n, p in param_dict.items() if 'v_proj' in n or 'o_proj' in n]
router_param = [p for n, p in param_dict.items() if 'gate.weight' in n]
mlp_param = [p for n, p in param_dict.items() if 'w1.weight' in n or 'w2.weight' in n or 'w3.weight' in n]
ln_param = [p for n, p in param_dict.items() if 'input_layernorm.weight' in n or 'post_attention_layernorm.weight' in n]
head_param = [p for n, p in param_dict.items() if 'lm_head.weight' in n]
rmsn_param = [p for n, p in param_dict.items() if 'model.norm.weight' in n]

# Create AdamW optimizer and use the fused version if it is available
optimizer = torch.optim.AdamW([
    {'params': embed_param, "name": "embed"},
    {'params': head_param, "name": "head"},
    {'params': ln_param, "name": "ln"},
    {'params': qk_param, "name": "qk"},
    {'params': vo_param, "name": "vo"},
    {'params': mlp_param, "name": "mlp"},
    {'params': router_param, "name": "router"},
    {'params': rmsn_param, "name": "rmsn"},
], lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits = model(X).logits.view(batch_size*block_size, -1)
                loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, min_lr, max_lr, alpha, warmup_iters, lr_decay_iters):
    warmup_iters *= alpha
    max_lr *= alpha
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
iter_num = 0
t0 = time.time()
raw_model = model.module if ddp else model # unwrap DDP container if needed
while True:
    # determine and set the learning rate for this iteration
    alphas = {
        "embed": embed_alpha,
        "head": head_alpha,
        "ln": ln_alpha,
        "qk": qk_alpha,
        "vo": vo_alpha,
        "mlp": mlp_alpha,
        "router": router_alpha,
        "rmsn": rmsn_alpha,
    }
    lrs = {}
    for param_group in optimizer.param_groups:
        group_name = param_group["name"]
        if group_name == "embed":
            lr = get_lr(iter_num, min_lr, learning_rate * alphas[group_name], 1., warmup_iters, lr_decay_iters)
        else:
            lr = get_lr(iter_num, min_lr, learning_rate, alphas[group_name], warmup_iters, lr_decay_iters)
        param_group['lr'] = lr
        lrs[group_name] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                **{f"lr/{name}_lr": lr for name, lr in lrs.items()},
            }, step=iter_num)
    
    if iter_num % save_interval == 0 and iter_num > 0 and master_process:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iter_num': iter_num,
            'loss': losses['val'],
            'config': config,
        }
        print(f"saving checkpoint to {out_dir}")
        os.makedirs(out_dir, exist_ok=True)
        torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))
    
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    model.require_backward_grad_sync = True
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            output = model(X)
        logits = output.logits.view(batch_size*block_size, -1)
        if output.aux_loss is not None:
            aux_loss = router_aux_loss_coef * output.aux_loss
            loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1) + aux_loss
        else:
            loss = F.cross_entropy(logits, Y.view(-1), ignore_index=-1)
        loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass
        loss.backward()
    # clip the gradient
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer
    optimizer.step()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
    iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
