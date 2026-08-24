"""True top-k Mixture-of-Experts, routed by a named ``expert`` axis.

Keeps only the ``k`` largest gates per token (via named ``topk`` + ``where``),
renormalizes, and mixes experts over the named ``E`` axis with ``unstack`` —
no dim=-1, no manual scatter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .. import structural as sp
from ..dim import Dim
from ..nn.layers import Linear, SwiGLU
from ..ops import reduce, softmax, sum
from ..tensor import NamedTensor


@dataclass
class MoEConfig:
    d_model: int = 32
    d_ff: int = 64
    n_experts: int = 8
    n_active: int = 2


class TopKMoE(nn.Module):
    """Softmax router → keep top-k experts per token → renormalize → mix."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.C = Dim("C", cfg.d_model)
        self.ff = Dim("ff", cfg.d_ff)
        self.E = Dim("E", cfg.n_experts)
        self.router = Linear(self.C, self.E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(self.C, self.ff) for _ in range(cfg.n_experts)])

    def forward(self, x: NamedTensor) -> NamedTensor:
        gates = softmax(self.router(x), over=self.E)          # (..., E)
        # kth-largest gate per token → threshold mask over the named E axis
        K = Dim("K")
        topv, _ = sp.topk(gates, self.E, k=self.cfg.n_active, kdim=K)
        kth = reduce(topv, over=K, how="min")                 # (...) no E
        keep = sp.ge(gates, kth)                              # bool (..., E)
        masked = sp.where(keep, gates, sp.zeros_like(gates))
        masked = masked / (sum(masked, over=self.E) + 1e-9)   # renormalize kept experts

        acc = sp.zeros_like(x)
        for w, expert in zip(sp.unstack(masked, self.E), self.experts):
            acc = acc + w * expert(x)  # w:(...) broadcasts over C
        return acc
