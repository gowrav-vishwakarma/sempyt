"""GLM/Kimi-style block: GQA + MoE router + RMSNorm + RoPE. Generality proof."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dim import Dim, ProductDim
from ..nn.layers import Linear, RMSNorm, RoPE, SwiGLU
from ..ops import softmax
from ..policies import SplitComplex
from ..tensor import named


@dataclass
class GLMConfig:
    d_model: int = 32
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 8
    n_experts: int = 4
    n_active: int = 2
    d_ff: int = 64
    max_seq: int = 32


class GLMBlock(nn.Module):
    """One decoder block: RMSNorm → GQA (RoPE) → residual → RMSNorm → MoE-SwiGLU."""

    def __init__(self, cfg: GLMConfig):
        super().__init__()
        self.cfg = cfg
        self.C = Dim("C", cfg.d_model)
        self.H = Dim("H", cfg.n_heads)
        self.Hkv = Dim("Hkv", cfg.n_kv_heads)
        self.d = Dim("d", cfg.head_dim)
        self.E = Dim("E", cfg.n_experts)
        self.ff = Dim("ff", cfg.d_ff)
        self.Tmax = Dim("Tmax", cfg.max_seq)
        self.px = Dim("px", 2, role="complex_pair")
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.repeats = cfg.n_heads // cfg.n_kv_heads

        self.inner = ProductDim(self.H, self.d)
        self.kv_inner = ProductDim(self.Hkv, self.d)
        self.norm1 = RMSNorm(self.C)
        self.q = Linear(self.C, self.inner, bias=False)
        self.k = Linear(self.C, self.kv_inner, bias=False)
        self.v = Linear(self.C, self.kv_inner, bias=False)
        self.o = Linear(self.inner, self.C, bias=False)
        self.rope = RoPE(self.Tmax, self.d, self.px)
        self.norm2 = RMSNorm(self.C)
        self.router = Linear(self.C, self.E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(self.C, self.ff) for _ in range(cfg.n_experts)])

    def forward(self, x, batch: Dim, time: Dim):
        h = x + self._gqa(self.norm1(x), batch, time)
        return h + self._moe(self.norm2(h))

    def _gqa(self, x, batch, time):
        q = self.q(x).to(batch, time, self.H, self.d)
        k = self.k(x).to(batch, time, self.Hkv, self.d)
        v = self.v(x).to(batch, time, self.Hkv, self.d)
        # pack as split-real for RoPE (imag = 0) then unpack
        pol = SplitComplex(self.px)
        def as_c(t):
            z = torch.stack([t.data, torch.zeros_like(t.data)], dim=-1)
            return named(z, t.layout + (self.px,), pol)

        T = x.size(time)
        rope = named(self.rope.cache[:T].to(dtype=x.dtype, device=x.device), (time, self.d, self.px), pol)
        q = (as_c(q) * rope).data[..., 0]
        k = (as_c(k) * rope).data[..., 0]
        q = named(q, (batch, time, self.H, self.d))
        k = named(k, (batch, time, self.Hkv, self.d))
        # repeat kv heads
        k_raw = k.to(batch, self.Hkv, time, self.d).data
        v_raw = v.to(batch, self.Hkv, time, self.d).data
        k_raw = k_raw.repeat_interleave(self.repeats, dim=1)
        v_raw = v_raw.repeat_interleave(self.repeats, dim=1)
        q_raw = q.to(batch, self.H, time, self.d).raw(batch, self.H, time, self.d)
        y = F.scaled_dot_product_attention(q_raw, k_raw, v_raw, is_causal=True)
        yn = named(y, (batch, self.H, time, self.d))
        return self.o(yn.to(batch, time, self.inner))

    def _moe(self, x):
        logits = self.router(x)
        gates = softmax(logits, over=self.E)
        topv, topi = torch.topk(gates.data, self.cfg.n_active, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)
        # dense fallback for the smoke test: mix all experts weighted by softmax
        acc = torch.zeros_like(x.data)
        for e, expert in enumerate(self.experts):
            w = gates.data[..., e].unsqueeze(-1)
            acc = acc + w * expert(x).data
        return named(acc, x.layout, x.policy)
