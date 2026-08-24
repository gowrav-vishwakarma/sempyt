"""GLM/Kimi-style block: GQA + MoE router + RMSNorm + RoPE. Generality proof."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from .. import structural as sp
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
        # pack a zero imaginary part as a NEW named 'px' axis, then RoPE-cmul
        pol = SplitComplex(self.px)

        def as_c(t):
            return sp.stack([t, sp.zeros_like(t)], into=self.px).with_policy(pol)

        T = x.size(time)
        rope = named(self.rope.cache[:T].to(dtype=x.dtype, device=x.device), (time, self.d, self.px), pol)
        q = sp.select(as_c(q) * rope, self.px, 0)  # real part after complex rotation
        k = sp.select(as_c(k) * rope, self.px, 0)
        # expand kv heads to full head count by naming the axis, not dim=1
        k = sp.repeat_along(k, self.Hkv, times=self.repeats, new=self.H)
        v = sp.repeat_along(v, self.Hkv, times=self.repeats, new=self.H)
        q_raw = q.raw(batch, self.H, time, self.d)
        k_raw = k.raw(batch, self.H, time, self.d)
        v_raw = v.raw(batch, self.H, time, self.d)
        y = F.scaled_dot_product_attention(q_raw, k_raw, v_raw, is_causal=True)
        yn = named(y, (batch, self.H, time, self.d))
        return self.o(yn.to(batch, time, self.inner))

    def _moe(self, x):
        # dense soft mixture over experts, addressed by the named E axis (no dim=-1).
        gates = softmax(self.router(x), over=self.E)
        acc = sp.zeros_like(x)
        for w, expert in zip(sp.unstack(gates, self.E), self.experts):
            acc = acc + w * expert(x)  # w:(batch,time) broadcasts over C
        return acc
