from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


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

        self.c_proj.RESIDUAL_SCALE_INIT = 1

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (
            1 / math.sqrt(hs)
        )  #  (B, nh, T, hs) @ ((B, nh, hs, T)) -> (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.RESIDUAL_SCALE_INIT = 1

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
            T < self.config.block_size
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


# =============================================================================


model = GPT.from_pretrained("gpt2")
print("GPT-2 pretrained model loaded successfully")


model.eval()
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model.to(device)

num_return_sequences = 5
max_length = 30

import tiktoken

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.shape[1] < max_length:
    with torch.no_grad():
        logits = model(x)  # (B,T,n_embd)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, num_samples=1)
        idx = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, idx), dim=1)

for i in range(num_return_sequences):
    decoded = enc.decode(x[i].tolist())
    print(f"> {decoded}")
