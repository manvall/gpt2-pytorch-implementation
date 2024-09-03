from dataclasses import dataclass
import inspect
import math
from typing import Callable
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import tiktoken
import os
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np

log_file = "./metrics.log"

# torch.set_float32_matmul_precision("high")

encoder = tiktoken.get_encoding("gpt2")

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:

    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ.get("RANK"), 0)  # type: ignore
    ddp_local_rank = int(os.environ.get("LOCAL_RANK"), 0)  # type: ignore
    ddp_world_size = int(os.environ["WORLD_SIZE"], 1)  # type: ignore
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device=}")

master_process = ddp_rank == 0
device_type = "cuda" if device.startswith("cuda") else "cpu"

# GPT-3 uses 0.5M tokens in a total batch.
# This represents the total number of tokens to be processed before an optimizer step, not the number of data rows.
# We accumulate gradients over mini-batches and only update the model weights once we've processed total_batch_size tokens.
total_batch_size = 256
B, T = 4, 32

assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "total_batch_size should be divisible by (B * T)."

grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"gradient accumulation steps: {grad_accum_steps}")

val_epochs = 100
val_loss_steps = 10
warm_up_steps = 15
sample_generate_epoch = 10
max_lr = 3e-4
min_lr = max_lr * 0.1
max_steps = 50
checkpoint_epoch = 10
use_compile = True


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    block_size: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12


class CausalAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd
        )  # All q,k,v in single matrix for efficiency
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.c_proj.RESIDUAL_SCALE_INIT = 1  # type: ignore

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1  # type: ignore

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)

        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # Weights sharing between word embedding and final lm_head layer
        # (768, 50257) -> 768 * 50257 = 38,597,376 parameters each for wte and lm_head layers. By sharing, we save 38M params

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "RESIDUAL_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert (
            T <= self.config.block_size
        ), f"Can't forward input sequences of length {T} greater than block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T)
        tok_embd = self.transformer.wte(idx)  # (B,T) -> (B, T, n_embd)
        pos_embd = self.transformer.wpe(pos)  # (T) ->    (T, n_embd)

        x = tok_embd + pos_embd

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]

        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [key for key in sd_keys if not key.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            key
            for key in sd_keys_hf
            if not (key.endswith(".attn.masked_bias") or key.endswith(".attn.bias"))
        ]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]

        # OpenAI checkpoints use a "Conv1D" module, but we use a vanilla Linear
        # So we have to transpose these weights when we import them

        # for k1, k2 in zip(sd_keys_hf, sd_keys):
        #     print(f"hf={k1}, our={k2}")

        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for key in sd_keys_hf:
            if any(key.endswith(w) for w in transposed):
                # print(f"{sd_hf[key].shape=}, {sd[key].shape=}")
                assert sd_hf[key].shape[::-1] == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key].t())
            else:
                assert sd_hf[key].shape == sd[key].shape
                with torch.no_grad():
                    sd[key].copy_(sd_hf[key])

        return model

    def configure_optimizers(self, weight_decay, lr, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params, no_decay_params = [], []
        for _, p in param_dict.items():
            if p.dim() >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_no_decay_params = sum(p.numel() for p in no_decay_params)

        if master_process:
            print(
                f"number of decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"number of non-decayed parameter tensors: {len(no_decay_params)}, with {num_no_decay_params:,} parameters"
            )

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters  # type: ignore
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)  # type: ignore
        return optimizer


# =============================================================================


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        assert split in {"train", "val"}

        # get the shard filenames
        data_root = "wikitext-small"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"found {len(shards)} shards for split {split}")

        # state, init at shard zero
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

        assert (
            len(self.tokens) > self.B * self.T
        ), "Current shard doesn't have enoug tokens for one batch"

    def next_batch(self):
        B, T = self.B, self.T

        batch = self.tokens[self.current_position : self.current_position + (B * T) + 1]
        x = batch[:-1].view(B, T)
        y = batch[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            # If next batch is out of bounds, go to next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y


def train_model(
    model: DDP | GPT,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_loader: DataLoaderLite,
    lr_scheduler: Callable,
    max_steps: int,
    grad_accum_steps: int,
    val_loader: DataLoaderLite | None = None,
):
    t0 = time.time()
    for step in range(max_steps):
        optimizer.zero_grad()
        loss_accum = 0.0

        val_loss = "NA"

        if step % val_epochs == 0 and val_loader:
            val_loss = eval_model(model, val_loader, val_loss_steps)
            model.train()  # Reset model to train mode
            if step > 0 and (step % checkpoint_epoch == 0 or step == max_steps):
                save_model(model, step, val_loss)

        if step % sample_generate_epoch == 0:
            generate_samples(model)
            model.train()

        for micro_step in range(grad_accum_steps):
            start = time.time()
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            if ddp:
                # Instead of syncing gradients across GPUs in ddp for each micro_step,
                # accumulate grads within each gpu first and sync the gradients only at the end of micro batches
                model.require_backward_grad_sync = micro_step == (grad_accum_steps - 1)  # type: ignore

            # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            # autocasting to bfloat16 on T4 is reducing the performance
            logits, loss = model(x, y)

            loss = loss / grad_accum_steps
            # Adjust the loss to account for gradient accumulation.
            # Instead of back propagating the full loss after each batch, we divide by the number of accumulation steps.
            # This ensures that the gradients are averaged over multiple batches, simulating a larger batch size.

            loss_accum += loss.detach()  # Detach tensor for loss tracking calculations
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        lr = lr_scheduler(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()

        if device_type == "cuda":
            torch.cuda.synchronize()

        duration = time.time() - start
        tokens_per_sec = (
            train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        ) / duration

        if step % 100 == 0 and master_process:
            print(
                f"Step {step}| loss: {loss_accum}  | val_loss: {val_loss}  | norm: {norm:.4f}  | time_taken(ms): {duration*1000:.2f}  | rate: {tokens_per_sec:.2f}tokens/sec"
            )
            with open(log_file, "a") as f:
                f.write(
                    f"Step {step}| loss: {loss_accum}  | val_loss: {val_loss}  | norm: {norm:.4f}  | time_taken(ms): {duration*1000:.2f}  | rate: {tokens_per_sec:.2f}tokens/sec"
                )

    if master_process:
        print(f"total_time: {time.time()-t0}")


def save_model(model, step, val_loss):
    # optionally write model checkpoints
    checkpoint_path = f"model_{step:05d}.pt"
    checkpoint = {
        "model": model.module.state_dict(),
        "config": raw_model.config,
        "step": step,
        "val_loss": val_loss.item(),
    }
    torch.save(checkpoint, checkpoint_path)


def eval_model(
    model: DDP | GPT,
    val_loader: DataLoaderLite,
    val_loss_steps: int,
):
    model.eval()  # Set model to eval mode
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = 0.0
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            x, y = x.to(device), y.to(device)

            _, loss = model(x, y)
            loss = loss / val_loss_steps

            val_loss_accum += loss.detach()
    if ddp:
        dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

    return val_loss_accum


def generate_samples(model):
    model.eval()
    num_return_sequences = 5
    max_length = 30

    tokens = encoder.encode("Hello, I'm a language model,")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    sample_x = tokens.to(device)

    rng_generator = torch.Generator(device=device)
    rng_generator.manual_seed(42 + ddp_rank)

    if device_type == "cuda":
        torch.cuda.manual_seed(42 + ddp_rank)

    while sample_x.shape[1] < max_length:
        with torch.no_grad():
            logits, _ = model(sample_x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, num_samples=1, generator=rng_generator)
            idx = torch.gather(topk_indices, -1, ix)
            sample_x = torch.cat((sample_x, idx), dim=1)

    for i in range(num_return_sequences):
        decoded = encoder.decode(sample_x[i].tolist())
        print(f"Rank {ddp_rank}> sample {i}: {decoded}")


def get_lr(step):
    if step < warm_up_steps:
        return max_lr * ((step + 1) / warm_up_steps)  # step + 1 to skip lr=0 at step=0
    elif step > max_steps:
        return min_lr

    decay_ratio = (step - warm_up_steps) / (max_steps - warm_up_steps)
    assert 0 <= decay_ratio <= 1
    scale = 0.5 * (1 + math.cos(decay_ratio * math.pi))  # 1 -> 0

    return (scale * (max_lr - min_lr)) + min_lr


# ===========================================================================

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
if use_compile:
    torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = (
    model.module if ddp else model
)  # ddpmodel.module contains the "raw" unwrapped model


with open("input.txt", "r") as f:
    text = f.read()


train_loader = DataLoaderLite(
    B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train"
)
val_loader = DataLoaderLite(
    B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val"
)

optimizer = raw_model.configure_optimizers(weight_decay=0.1, lr=max_lr, device_type=device_type)  # type: ignore

train_model(model, optimizer, train_loader, get_lr, max_steps=20, grad_accum_steps=2)


if ddp:
    destroy_process_group()

# =============================================================================
