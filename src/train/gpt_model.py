# gpt_model.py
import math
import torch
import torch.nn as nn


class GPTConfig:
    def __init__(self, vocab_size, n_ctx, n_layer, d_model, n_head, d_ff):
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_head = n_head
        self.d_ff = d_ff


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.n_ctx, config.d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        device = idx.device

        tok = self.token_emb(idx)
        positions = torch.arange(t, device=device).unsqueeze(0)
        pos = self.pos_emb(positions)

        x = tok + pos

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = targets.reshape(-1)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)


        return logits, loss


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)

        # -----------------------------
        # GELU instead of ReLU here
        # -----------------------------
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_head == 0

        self.n_head = config.n_head
        self.d_head = config.d_model // config.n_head

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(torch.ones(config.n_ctx, config.n_ctx))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        causal_mask = self.mask[:T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)
