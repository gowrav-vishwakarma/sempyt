"""RWKV-style block (time-mix WKV + channel-mix), written with named axes.

Token-shift is a named ``shift`` over ``time``; the WKV operator is two named
scans (numerator/denominator) with a constant per-channel decay. No dim=.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .. import structural as sp
from ..dim import Dim
from ..nn.layers import LayerNorm, Linear
from ..ops import scan
from ..tensor import NamedTensor, named


@dataclass
class RWKVConfig:
    d_model: int = 32
    d_ffn: int = 64
    scan_mode: str = "recurrent"


def _mix(x, xs, mu, C):
    m = named(mu, (C,))
    one = named(torch.ones_like(mu), (C,))
    return x * m + xs * (one - m)


class RWKVTimeMix(nn.Module):
    def __init__(self, cfg: RWKVConfig, C: Dim):
        super().__init__()
        self.C = C
        self.cfg = cfg
        self.key = Linear(C, C, bias=False)
        self.value = Linear(C, C, bias=False)
        self.recept = Linear(C, C, bias=False)
        self.out = Linear(C, C, bias=False)
        self.time_decay = nn.Parameter(torch.zeros(C.size))
        self.mu_k = nn.Parameter(torch.rand(C.size))
        self.mu_v = nn.Parameter(torch.rand(C.size))
        self.mu_r = nn.Parameter(torch.rand(C.size))

    def forward(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        xs = sp.shift(x, time, by=1)  # previous token, zero at t=0
        k = self.key(_mix(x, xs, self.mu_k, self.C))
        v = self.value(_mix(x, xs, self.mu_v, self.C))
        r = sp.sigmoid(self.recept(_mix(x, xs, self.mu_r, self.C)))

        lam = named(torch.exp(-torch.exp(self.time_decay)), (self.C,))  # decay in (0,1)
        ek = sp.exp(sp.clamp(k, max=30.0))
        num = scan(ek * v, lam, over=time, mode=self.cfg.scan_mode)
        den = scan(ek, lam, over=time, mode=self.cfg.scan_mode)
        wkv = num / (den + 1e-6)
        return self.out(r * wkv)


class RWKVChannelMix(nn.Module):
    def __init__(self, cfg: RWKVConfig, C: Dim, ff: Dim):
        super().__init__()
        self.C = C
        self.ff = ff
        self.key = Linear(C, ff, bias=False)
        self.value = Linear(ff, C, bias=False)
        self.recept = Linear(C, C, bias=False)
        self.mu_k = nn.Parameter(torch.rand(C.size))
        self.mu_r = nn.Parameter(torch.rand(C.size))

    def forward(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        xs = sp.shift(x, time, by=1)
        k = sp.square(sp.relu(self.key(_mix(x, xs, self.mu_k, self.C))))
        r = sp.sigmoid(self.recept(_mix(x, xs, self.mu_r, self.C)))
        return r * self.value(k)


class RWKVBlock(nn.Module):
    def __init__(self, cfg: RWKVConfig):
        super().__init__()
        self.C = Dim("C", cfg.d_model)
        self.ff = Dim("ff", cfg.d_ffn)
        self.ln1 = LayerNorm(self.C)
        self.ln2 = LayerNorm(self.C)
        self.time_mix = RWKVTimeMix(cfg, self.C)
        self.chan_mix = RWKVChannelMix(cfg, self.C, self.ff)

    def forward(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        x = x + self.time_mix(self.ln1(x), batch, time)
        return x + self.chan_mix(self.ln2(x), batch, time)
