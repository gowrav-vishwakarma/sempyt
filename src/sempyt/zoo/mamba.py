"""Mamba-style selective SSM block, written entirely with named axes.

State-space recurrence ``h_t = exp(Δ_t·A) ⊙ h_{t-1} + (Δ_t·B_t)·x_t`` per
channel, read out with ``y_t = Σ_state C_t ⊙ h_t``. The scan runs over the
named ``time`` axis; the state lives on a named ``state`` axis. No dim= anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .. import structural as sp
from ..dim import Dim
from ..nn.layers import Linear
from ..ops import contract, scan
from ..tensor import NamedTensor, named


@dataclass
class MambaConfig:
    d_model: int = 32
    d_inner: int = 64
    d_state: int = 16
    scan_mode: str = "recurrent"  # recurrent | chunked | parallel


class MambaBlock(nn.Module):
    """One selective-scan block. Channels = ``inner``; SSM state = ``state``."""

    def __init__(self, cfg: MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.C = Dim("C", cfg.d_model)
        self.inner = Dim("inner", cfg.d_inner)
        self.state = Dim("state", cfg.d_state)

        self.in_proj = Linear(self.C, self.inner, bias=False)
        self.gate_proj = Linear(self.C, self.inner, bias=False)
        self.dt_proj = Linear(self.inner, self.inner, bias=True)
        self.B_proj = Linear(self.inner, self.state, bias=False)
        self.C_proj = Linear(self.inner, self.state, bias=False)
        self.out_proj = Linear(self.inner, self.C, bias=False)
        # A is negative (stable decay); stored as log for positivity of -A
        self.A_log = nn.Parameter(torch.log(torch.arange(1, cfg.d_state + 1).float()
                                             .expand(cfg.d_inner, cfg.d_state).clone()))

    def forward(self, x: NamedTensor, batch: Dim, time: Dim) -> NamedTensor:
        u = self.in_proj(x).to(batch, time, self.inner)          # (B,T,inner)
        z = self.gate_proj(x)                                     # gate
        # data-dependent step size Δ_t > 0
        delta = sp.softplus(self.dt_proj(u))                      # (B,T,inner)
        Bt = self.B_proj(u)                                       # (B,T,state)
        Ct = self.C_proj(u)                                       # (B,T,state)
        A = named(-torch.exp(self.A_log), (self.inner, self.state))

        # broadcast to (B,T,inner,state): per-step decay and input drive
        dA = delta * A                                            # (B,T,inner,state)
        lam = sp.exp(dA)                                          # decay in (0,1)
        drive = (delta * u) * Bt                                 # (B,T,inner,state)

        h = scan(drive, lam, over=time, mode=self.cfg.scan_mode)  # (B,T,inner,state)
        y = contract(h, Ct, over=self.state)                      # (B,T,inner)
        y = y * sp.silu(z)                                        # gated output
        return self.out_proj(y)
