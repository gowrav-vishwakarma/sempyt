"""BERT-style bidirectional encoder, written with named axes.

Same building blocks as the GPT zoo model but non-causal attention and a
learned segment/position embedding. Demonstrates that the named surface is not
decoder-only. No dim= in the model body.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..dim import Dim
from ..nn.layers import Embed, GLU, LayerNorm, SDPAAttention
from ..tensor import NamedTensor, named


@dataclass
class BertConfig:
    vocab_size: int = 30522
    max_seq_len: int = 512
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128


class BertLayer(nn.Module):
    def __init__(self, cfg: BertConfig, C: Dim, H: Dim, d: Dim, ff: Dim):
        super().__init__()
        self.attn = SDPAAttention(C, H, d, causal=False, bias=True)
        self.ln1 = LayerNorm(C)
        self.ffn = GLU(C, ff, act=torch.nn.functional.gelu)
        self.ln2 = LayerNorm(C)

    def forward(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        x = self.ln1(x + self.attn(x, time=time, batch=batch))
        return self.ln2(x + self.ffn(x))


class BertEncoder(nn.Module):
    """Bidirectional transformer encoder returning contextual hidden states."""

    def __init__(self, cfg: BertConfig):
        super().__init__()
        self.cfg = cfg
        self.C = Dim("C", cfg.d_model)
        self.H = Dim("H", cfg.n_heads)
        self.d = Dim("d", cfg.d_model // cfg.n_heads)
        self.ff = Dim("ff", cfg.d_ff)
        self.V = Dim("V", cfg.vocab_size)
        self.P = Dim("P", cfg.max_seq_len)
        self.tok = Embed(self.V, self.C)
        self.pos = Embed(self.P, self.C)
        self.ln0 = LayerNorm(self.C)
        self.layers = nn.ModuleList(
            [BertLayer(cfg, self.C, self.H, self.d, self.ff) for _ in range(cfg.n_layers)]
        )

    def forward(self, input_ids: torch.Tensor, batch: Dim, time: Dim) -> NamedTensor:
        T = input_ids.shape[1]
        pos_ids = torch.arange(T, device=input_ids.device).expand_as(input_ids)
        x = self.tok(input_ids, batch, time) + self.pos(pos_ids, batch, time)
        x = self.ln0(x)
        for layer in self.layers:
            x = layer(x, batch, time)
        return x
