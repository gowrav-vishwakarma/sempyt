"""pands v1 outer-product memory + v2 diag_scan, written with named axes."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..dim import Dim, ProductDim
from ..nn.layers import Linear, PhaseLUT
from ..ops import contract, outer, scan
from ..policies import SplitComplex
from ..tensor import NamedTensor, named


def _strict_decay_matrix(gamma: torch.Tensor, seq_len: int) -> torch.Tensor:
    device = gamma.device
    idx = torch.arange(seq_len, device=device)
    exponent = (idx.unsqueeze(-1) - 1 - idx.unsqueeze(-2)).float()
    mask = idx.unsqueeze(-1) > idx.unsqueeze(-2)
    exponent = torch.where(mask, exponent, torch.zeros_like(exponent))
    log_g = torch.log(gamma.clamp_min(1e-6)).view(-1, 1, 1)
    d = torch.exp(exponent.unsqueeze(0) * log_g)
    return torch.where(mask.unsqueeze(0), d, torch.zeros_like(d))


class OuterProductBand(nn.Module):
    """One band: S += v⊗k, y = S@q, exclusive read. Three modes agree."""

    def __init__(self, dim: Dim, heads: Dim, head_dim: Dim, decay_bias: float = -2.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim
        self.di = Dim(f"{head_dim.name}_i", head_dim.size)
        self.dj = Dim(f"{head_dim.name}_j", head_dim.size)
        self.inner = ProductDim(heads, head_dim)
        self.q = Linear(dim, self.inner, bias=False)
        self.k = Linear(dim, self.inner, bias=False)
        self.v = Linear(dim, self.inner, bias=False)
        self.o = Linear(self.inner, dim, bias=False)
        self.decay_bias = nn.Parameter(torch.full((heads.size,), decay_bias))

    def _qkv(self, x: NamedTensor, time: Dim, batch: Dim):
        H, d = self.heads, self.head_dim
        q = self.q(x).to(batch, H, time, d) * (d.size ** -0.5)
        k = self.k(x).to(batch, H, time, d)
        v = self.v(x).to(batch, H, time, d)
        gamma = torch.sigmoid(self.decay_bias)
        return q, k, v, gamma

    def recurrent(self, x: NamedTensor, time: Dim, batch: Dim, state=None):
        H, d = self.heads, self.head_dim
        q, k, v, gamma = self._qkv(x, time, batch)
        B, T = x.size(batch), x.size(time)
        if state is None:
            state = named(
                x.data.new_zeros(B, H.size, d.size, d.size),
                (batch, H, self.di, self.dj),
            )
        g = named(gamma.view(1, H.size).expand(B, H.size), (batch, H))
        outs = []
        for t in range(T):
            qt = named(q.data[:, :, t], (batch, H, d)).alias(d, self.dj)
            kt = named(k.data[:, :, t], (batch, H, d)).alias(d, self.dj)
            vt = named(v.data[:, :, t], (batch, H, d)).alias(d, self.di)
            outs.append(contract(state, qt, over=self.dj).data)
            state = state * g + outer(vt, kt, (self.di, self.dj))
        y = named(torch.stack(outs, dim=2), (batch, H, time, self.di))
        return self.o(y.alias(self.di, d).to(batch, time, self.inner)), state

    def parallel(self, x: NamedTensor, time: Dim, batch: Dim):
        H, d = self.heads, self.head_dim
        q, k, v, gamma = self._qkv(x, time, batch)
        T = x.size(time)
        Ts, Tk = Dim("Ts", T), Dim("Tk", T)
        decay = named(_strict_decay_matrix(gamma, T).to(dtype=x.dtype), (H, Ts, Tk))
        scores = contract(q.alias(time, Ts), k.alias(time, Tk), over=d)
        local = contract(scores * decay, v.alias(time, Tk), over=Tk)
        idx = torch.arange(T, device=x.device)
        g_pow = torch.exp(
            (T - 1 - idx).float().view(1, 1, T)
            * torch.log(gamma.clamp_min(1e-6)).view(1, H.size, 1)
        )
        vk = v.data * g_pow.unsqueeze(-1)
        final = torch.einsum("bhti,bhtj->bhij", vk, k.data)
        y = self.o(local.alias(Ts, time).to(batch, time, self.inner))
        return y, named(final, (batch, H, self.di, self.dj))

    def chunked(self, x: NamedTensor, time: Dim, batch: Dim, chunk: int = 128):
        if chunk <= 0 or chunk >= x.size(time):
            return self.parallel(x, time, batch)
        H, d = self.heads, self.head_dim
        q, k, v, gamma = self._qkv(x, time, batch)
        T = x.size(time)
        log_g = torch.log(gamma.clamp_min(1e-6))
        state = named(
            x.data.new_zeros(x.size(batch), H.size, d.size, d.size),
            (batch, H, self.di, self.dj),
        )
        outputs = []
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            c = end - start
            Tc = Dim("Tc", c)
            Ts, Tk = Dim("Ts", c), Dim("Tk", c)
            qc = named(q.data[:, :, start:end], (batch, H, Tc, d)).alias(Tc, Ts)
            kc = named(k.data[:, :, start:end], (batch, H, Tc, d)).alias(Tc, Tk)
            vc = named(v.data[:, :, start:end], (batch, H, Tc, d)).alias(Tc, Tk)
            decay = named(_strict_decay_matrix(gamma, c).to(dtype=x.dtype), (H, Ts, Tk))
            local = contract(contract(qc, kc, over=d) * decay, vc, over=Tk)
            idx = torch.arange(c, device=x.device, dtype=torch.float32)
            q_for_s = named(q.data[:, :, start:end], (batch, H, Tc, d)).alias(d, self.dj)
            carry = contract(state, q_for_s, over=self.dj).to(batch, H, Tc, self.di)
            g_i = torch.exp(idx.view(1, 1, c, 1) * log_g.view(1, H.size, 1, 1))
            carry = named(carry.data * g_i, (batch, H, Tc, self.di))
            loc = local.alias(Ts, Tc).to(batch, H, Tc, d)
            outputs.append(loc.data + carry.alias(self.di, d).data)
            g_pow = torch.exp((c - 1 - idx).view(1, 1, c) * log_g.view(1, H.size, 1))
            state_chunk = torch.einsum(
                "bhti,bhtj->bhij",
                v.data[:, :, start:end] * g_pow.unsqueeze(-1),
                k.data[:, :, start:end],
            )
            g_c = torch.exp(log_g * c).view(1, H.size, 1, 1)
            state = named(g_c * state.data + state_chunk, state.layout)
        y = named(torch.cat(outputs, dim=2), (batch, H, time, self.di))
        return self.o(y.alias(self.di, d).to(batch, time, self.inner)), state


def diag_scan_named(
    u: NamedTensor,
    lam: NamedTensor,
    time: Dim,
    mode: str = "chunked",
    chunk: int = 64,
) -> NamedTensor:
    """Inclusive ``h_t = λ ⊙ h_{t-1} + u_t``. ``u`` has ``time``; ``lam`` does not."""
    return scan(u, lam, over=time, mode=mode, chunk=chunk)
