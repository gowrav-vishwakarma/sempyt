"""GPT-2-style LM written with named axes. Parameter names match pands TransformerLM."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import structural as sp
from ..dim import Dim, ProductDim
from ..tensor import named


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, dims):
        super().__init__()
        C, H, d = dims["C"], dims["H"], dims["d"]
        assert C.size == H.size * d.size
        self.n_heads = config.n_heads
        self.head_dim = config.n_heads and (config.d_model // config.n_heads)
        self.C, self.H, self.d = C, H, d
        self.inner = ProductDim(H, d)
        self.qkv = nn.Linear(C.size, 3 * C.size)
        self.out_proj = nn.Linear(C.size, C.size)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.slot = Dim("slot", 3)

    def forward(self, x):
        B, T, C = x.layout
        qkv = F.linear(x.raw(B, T, C), self.qkv.weight, self.qkv.bias)
        qkv3 = Dim("qkv3", 3 * C.size)
        # split the packed projection along the named 'slot' axis (q/k/v)
        packed = (
            named(qkv, (B, T, qkv3))
            .to(B, T, self.slot, self.H, self.d)
            .to(B, self.slot, self.H, T, self.d)
        )
        q, k, v = sp.unstack(packed, self.slot)  # each (B, H, T, d)
        drop = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(
            q.raw(B, self.H, T, self.d),
            k.raw(B, self.H, T, self.d),
            v.raw(B, self.H, T, self.d),
            is_causal=True,
            dropout_p=drop,
        )
        yn = named(y, (B, self.H, T, self.d)).to(B, T, self.inner)
        out = F.linear(yn.data, self.out_proj.weight, self.out_proj.bias)
        return named(self.resid_dropout(out), (B, T, C))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.layout
        h = self.dropout(self.fc2(F.gelu(self.fc1(x.raw(B, T, C)))))
        return named(h, (B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig, dims):
        super().__init__()
        C = dims["C"]
        self.ln1 = nn.LayerNorm(C.size)
        self.attn = CausalSelfAttention(config, dims)
        self.ln2 = nn.LayerNorm(C.size)
        self.ffn = FeedForward(config)
        self.C = C

    def forward(self, x):
        B, T, C = x.layout
        n1 = named(self.ln1(x.raw(B, T, C)), (B, T, C))
        x = x + self.attn(n1)
        n2 = named(self.ln2(x.raw(B, T, C)), (B, T, C))
        return x + self.ffn(n2)


class GPTLM(nn.Module):
    """Decoder-only GPT-2. ``state_dict`` keys match pands ``TransformerLM``."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.B = Dim("B")
        self.T = Dim("T")
        self.C = Dim("C", config.d_model)
        self.H = Dim("H", config.n_heads)
        self.d = Dim("d", config.d_model // config.n_heads)
        self.V = Dim("V", config.vocab_size)
        dims = {"C": self.C, "H": self.H, "d": self.d}

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config, dims) for _ in range(config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight
        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(p)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.n_layers),
            )
            nn.init.normal_(
                block.ffn.fc2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.n_layers),
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        Bsz, Tsz = input_ids.shape
        if self.B.size is None:
            self.B.bind(Bsz)
        if self.T.size is None:
            self.T.bind(Tsz)
        T = self.T if self.T.size == Tsz else Dim("T", Tsz)
        B = self.B if self.B.size == Bsz else Dim("B", Bsz)
        pos = torch.arange(0, Tsz, dtype=torch.long, device=input_ids.device)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(pos))
        xt = named(x, (B, T, self.C))
        for block in self.blocks:
            xt = block(xt)
        data = self.ln_f(xt.raw(B, T, self.C))
        return self.lm_head(data)
