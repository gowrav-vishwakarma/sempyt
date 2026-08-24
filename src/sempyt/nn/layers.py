"""Named-axis layers. Weights are plain nn.Parameter; forwards take NamedTensors."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..dim import Dim, ProductDim
from ..ops import Index, at, contract, outer, softmax
from ..policies import REAL, SplitComplex
from ..tensor import NamedTensor, named


class Linear(nn.Module):
    """``y = x @ W.T + b`` over a named in-axis. Matches ``nn.Linear`` numerics."""

    def __init__(self, inn: Dim, out: Dim, bias: bool = True):
        super().__init__()
        if inn.size is None or out.size is None:
            raise ValueError("Linear in/out Dims must be bound")
        self.inn = inn
        self.out = out
        self.weight = nn.Parameter(torch.empty(out.size, inn.size))
        self.bias = nn.Parameter(torch.empty(out.size)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(inn.size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: NamedTensor) -> NamedTensor:
        def _is_in(d):
            if d is self.inn:
                return True
            return [id(a) for a in d.leaves()] == [id(a) for a in self.inn.leaves()]

        rest = [d for d in x.layout if not _is_in(d)]
        xl = x.to(*rest, self.inn)
        y = F.linear(xl.data, self.weight, self.bias)
        return named(y, rest + [self.out], x.policy)


class RMSNorm(nn.Module):
    def __init__(self, dim: Dim, eps: float = 1e-6):
        super().__init__()
        if dim.size is None:
            raise ValueError("RMSNorm dim must be bound")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim.size))

    def forward(self, x: NamedTensor) -> NamedTensor:
        xl = x.to(*[d for d in x.layout if d is not self.dim], self.dim)
        var = xl.data.pow(2).mean(dim=-1, keepdim=True)
        y = xl.data * torch.rsqrt(var + self.eps) * self.weight
        return named(y, xl.layout, x.policy)


class LayerNorm(nn.Module):
    def __init__(self, dim: Dim, eps: float = 1e-5):
        super().__init__()
        if dim.size is None:
            raise ValueError("LayerNorm dim must be bound")
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim.size))
        self.bias = nn.Parameter(torch.zeros(dim.size))

    def forward(self, x: NamedTensor) -> NamedTensor:
        xl = x.to(*[d for d in x.layout if d is not self.dim], self.dim)
        y = F.layer_norm(xl.data, (self.dim.size,), self.weight, self.bias, self.eps)
        return named(y, xl.layout, x.policy)


class Embed(nn.Module):
    def __init__(self, vocab: Dim, dim: Dim):
        super().__init__()
        if vocab.size is None or dim.size is None:
            raise ValueError("Embed dims must be bound")
        self.vocab = vocab
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab.size, dim.size))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor, *id_layout: Dim) -> NamedTensor:
        data = F.embedding(ids, self.weight)
        return named(data, tuple(id_layout) + (self.dim,))


class FactorizedEmbed(nn.Module):
    """``ids → [..., rank] → [..., dim]`` — pands-style factorized vocab embed."""

    def __init__(self, vocab: Dim, dim: Dim, rank: Dim):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.rank = rank
        self.u = nn.Embedding(vocab.size, rank.size)
        self.v = nn.Linear(rank.size, dim.size, bias=False)

    def forward(self, ids: torch.Tensor, *id_layout: Dim) -> NamedTensor:
        data = self.v(self.u(ids))
        return named(data, tuple(id_layout) + (self.dim,))


class TiedHead(nn.Module):
    """Tied softmax head: ``hidden @ embed.weight.T``."""

    def __init__(self, embed: Embed):
        super().__init__()
        self.embed = embed

    def forward(self, x: NamedTensor) -> NamedTensor:
        xl = x.to(*[d for d in x.layout if d is not self.embed.dim], self.embed.dim)
        logits = F.linear(xl.data, self.embed.weight)
        return named(logits, xl.layout[:-1] + (self.embed.vocab,), x.policy)


class RoPE(nn.Module):
    """Rotary positions as a split-real cache ``[T, d, px]``, applied with cmul."""

    def __init__(self, time: Dim, dim: Dim, pair: Dim, base: float = 10000.0):
        super().__init__()
        self.time = time
        self.dim = dim
        self.pair = pair
        max_t = time.size
        d = dim.size
        if max_t is None or d is None:
            raise ValueError("RoPE time/dim must be bound")
        if pair.size is None:
            pair.bind(2)
        inv = 1.0 / (base ** (torch.arange(0, d, dtype=torch.float32) / d))
        pos = torch.arange(max_t, dtype=torch.float32)
        angles = pos[:, None] * inv[None, :]
        cache = torch.stack([angles.cos(), angles.sin()], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: NamedTensor, offset: int = 0) -> NamedTensor:
        T = x.size(self.time)
        rope = self.cache[offset : offset + T].to(dtype=x.dtype, device=x.device)
        rope_n = named(rope, (self.time, self.dim, self.pair), x.policy)
        return x * rope_n


class GLU(nn.Module):
    def __init__(self, dim: Dim, hidden: Dim, act=F.silu):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.act = act
        self.gate = Linear(dim, hidden, bias=False)
        self.up = Linear(dim, hidden, bias=False)
        self.down = Linear(hidden, dim, bias=False)

    def forward(self, x: NamedTensor) -> NamedTensor:
        return self.down(self.gate(x) * self.up(x))  # type: ignore[operator]


class SwiGLU(GLU):
    def __init__(self, dim: Dim, hidden: Dim):
        super().__init__(dim, hidden, act=F.silu)

    def forward(self, x: NamedTensor) -> NamedTensor:
        g = self.gate(x)
        # apply silu on the hidden axis (last after Linear)
        data = F.silu(g.data) * self.up(x).data
        gated = named(data, g.layout, x.policy)
        return self.down(gated)


class SDPAAttention(nn.Module):
    """Causal MHA via ``F.scaled_dot_product_attention``. ``.raw()`` at the kernel edge."""

    def __init__(
        self,
        dim: Dim,
        heads: Dim,
        head_dim: Dim,
        dropout: float = 0.0,
        causal: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.inner = ProductDim(heads, head_dim)
        self.qkv3 = Dim("qkv3", 3 * self.inner.size)
        self.slot = Dim("slot", 3)
        self.dropout = dropout
        self.causal = causal
        self.qkv = Linear(dim, self.qkv3, bias=bias)
        self.out_proj = Linear(self.inner, dim, bias=bias)

    def forward(self, x: NamedTensor, time: Dim, batch: Dim) -> NamedTensor:
        qkv = self.qkv(x)
        split = qkv.to(
            *[d for d in qkv.layout if d is not self.qkv3],
            self.slot,
            self.heads,
            self.head_dim,
        )
        raw = split.to(batch, self.slot, self.heads, time, self.head_dim).raw(
            batch, self.slot, self.heads, time, self.head_dim
        )
        q, k, v = raw[:, 0], raw[:, 1], raw[:, 2]
        drop = self.dropout if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=drop)
        yn = named(y, (batch, self.heads, time, self.head_dim), x.policy)
        return self.out_proj(yn.to(batch, time, self.inner))


class LinearAttention(nn.Module):
    """GLA-style: scores = q·k, weighted by a decay matrix, times v. Exclusive read."""

    def __init__(self, dim: Dim, heads: Dim, head_dim: Dim):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        inner = ProductDim(heads, head_dim)
        self.q = Linear(dim, inner, bias=False)
        self.k = Linear(dim, inner, bias=False)
        self.v = Linear(dim, inner, bias=False)
        self.o = Linear(inner, dim, bias=False)
        self.inner = inner
        self.decay_bias = nn.Parameter(torch.full((heads.size,), -2.0))

    def forward(self, x: NamedTensor, time: Dim, batch: Dim) -> NamedTensor:
        H, d = self.heads, self.head_dim
        q = self.q(x).to(batch, H, time, d) * (d.size ** -0.5)
        k = self.k(x).to(batch, H, time, d)
        v = self.v(x).to(batch, H, time, d)
        gamma = torch.sigmoid(self.decay_bias)
        T = x.size(time)
        decay = _strict_decay_matrix(gamma, T, x.data.device, x.dtype)  # [H,T,T]
        Ts = Dim("Ts", T)
        Tk = Dim("Tk", T)
        q2 = q.alias(time, Ts)
        k2 = k.alias(time, Tk)
        v2 = v.alias(time, Tk)
        scores = contract(q2, k2, over=d)  # B H Ts Tk
        D = named(decay, (H, Ts, Tk))
        local = contract(scores * D, v2, over=Tk)
        y = local.alias(Ts, time).to(batch, time, self.inner)
        return self.o(y)


def _strict_decay_matrix(gamma, seq_len, device, dtype):
    idx = torch.arange(seq_len, device=device)
    exponent = (idx.unsqueeze(-1) - 1 - idx.unsqueeze(-2)).float()
    mask = idx.unsqueeze(-1) > idx.unsqueeze(-2)
    exponent = torch.where(mask, exponent, torch.zeros_like(exponent))
    log_g = torch.log(gamma.clamp_min(1e-6)).view(-1, 1, 1)
    d = torch.exp(exponent.unsqueeze(0) * log_g)
    return torch.where(mask.unsqueeze(0), d, torch.zeros_like(d)).to(dtype=dtype)


class OuterProductMemory(nn.Module):
    """Write ``S += v ⊗ k``, read ``y = S @ q``. Exclusive (read before write)."""

    def __init__(self, dim: Dim, heads: Dim, head_dim: Dim, decay_bias: float = -2.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.di = Dim(f"{head_dim.name}_i", head_dim.size)
        self.dj = Dim(f"{head_dim.name}_j", head_dim.size)
        inner = ProductDim(heads, head_dim)
        self.q = Linear(dim, inner, bias=False)
        self.k = Linear(dim, inner, bias=False)
        self.v = Linear(dim, inner, bias=False)
        self.o = Linear(inner, dim, bias=False)
        self.inner = inner
        self.decay_bias = nn.Parameter(torch.full((heads.size,), decay_bias))

    def recurrent(
        self,
        x: NamedTensor,
        time: Dim,
        batch: Dim,
        state: Optional[NamedTensor] = None,
    ) -> tuple[NamedTensor, NamedTensor]:
        H, d = self.heads, self.head_dim
        q = self.q(x).to(batch, H, time, d) * (d.size ** -0.5)
        k = self.k(x).to(batch, H, time, d)
        v = self.v(x).to(batch, H, time, d)
        gamma = torch.sigmoid(self.decay_bias).view(1, H.size, 1, 1)
        B, T = x.size(batch), x.size(time)
        if state is None:
            sdata = x.data.new_zeros(B, H.size, d.size, d.size)
            state = named(sdata, (batch, H, self.di, self.dj))
        outs = []
        for t in range(T):
            qt = named(q.data[:, :, t], (batch, H, d), x.policy).alias(d, self.dj)
            kt = named(k.data[:, :, t], (batch, H, d), x.policy).alias(d, self.dj)
            vt = named(v.data[:, :, t], (batch, H, d), x.policy).alias(d, self.di)
            yt = contract(state, qt, over=self.dj)
            outs.append(yt.data)
            g = named(gamma.expand(B, H.size, 1, 1)[:, :, 0, 0], (batch, H))
            state = state * g + outer(vt, kt, (self.di, self.dj))
        y = named(torch.stack(outs, dim=2), (batch, H, time, self.di), x.policy)
        y = y.alias(self.di, d).to(batch, time, self.inner)
        return self.o(y), state

    def parallel(self, x: NamedTensor, time: Dim, batch: Dim) -> tuple[NamedTensor, NamedTensor]:
        return LinearAttention.forward(self._as_linattn(), x, time, batch), None  # type: ignore

    def _as_linattn(self):
        # reuse LinearAttention path — not a clean cast; callers use recurrent/zoo
        return self


class PhaseLUT(nn.Module):
    """Unit-circle table ``LUT[k] = (cos 2πk/N, sin 2πk/N)``. Buffer, not a Parameter."""

    ALLOWED = (64, 128, 256, 512, 1024, 4096)

    def __init__(self, bins: Dim, pair: Dim, phase_mode: str = "lut"):
        super().__init__()
        n = bins.size
        if n not in self.ALLOWED:
            raise ValueError(f"phase_bins must be one of {self.ALLOWED}, got {n}")
        if phase_mode not in ("lut", "lut_interp", "exact"):
            raise ValueError(f"unknown phase_mode {phase_mode}")
        self.bins = bins
        self.pair = pair
        if pair.size is None:
            pair.bind(2)
        self.phase_mode = phase_mode
        theta = 2.0 * math.pi * torch.arange(n, dtype=torch.float32) / n
        lut = torch.stack([theta.cos(), theta.sin()], dim=-1)
        self.register_buffer("lut", lut, persistent=False)

    def from_angle(self, theta: NamedTensor, mode: Optional[str] = None) -> NamedTensor:
        mode = mode or self.phase_mode
        if mode == "exact":
            data = torch.stack([theta.data.cos(), theta.data.sin()], dim=-1)
            return named(data, theta.layout + (self.pair,), SplitComplex(self.pair))
        n = self.bins.size
        scaled = theta.data * (n / (2.0 * math.pi))
        scaled = torch.remainder(scaled, n)
        if mode == "lut":
            k = torch.round(scaled).long() % n
            out = self.lut[k]
        else:
            k0 = torch.floor(scaled).long() % n
            k1 = (k0 + 1) % n
            frac = (scaled - torch.floor(scaled)).unsqueeze(-1)
            out = (1.0 - frac) * self.lut[k0] + frac * self.lut[k1]
        return named(out, theta.layout + (self.pair,), SplitComplex(self.pair))

    def rotate(self, z: NamedTensor, rot: NamedTensor) -> NamedTensor:
        return z * rot
