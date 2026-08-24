"""Vendored GPT-2-style reference baseline for offline equivalence testing.

Trimmed, self-contained copy of pands ``v1/transformer_baseline.py``. Only the
classes needed by ``tests/test_ref_transformer.py`` are kept, byte-for-byte in
behavior so ``state_dict`` keys and numerics match ``sempyt.zoo.gpt.GPTLM``.
Dropped: ``generate``, ``count_parameters``, ``get_transformer_config_100m``,
``TRANSFORMER_CONFIGS``.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int = 50257
    max_seq_len: int = 2048
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(y))


class FeedForward(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only transformer language model (GPT-2-style)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_weights:
            self.lm_head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        """GPT-2-style initialization."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
        for block in self.blocks:
            nn.init.normal_(
                block.attn.out_proj.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
            )
            nn.init.normal_(
                block.ffn.fc2.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers)
            )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (B, T, vocab_size)."""
        B, T = input_ids.size()
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        x = self.drop(self.token_embed(input_ids) + self.pos_embed(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)
