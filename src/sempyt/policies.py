"""Element-wise math policies. Complex algebra is a swap-in, not baked in."""

from __future__ import annotations

from typing import Optional

import torch

from .dim import Dim


class ElementPolicy:
    """mul / add / sub / neg / conj / abs / norm / exp. Override to change algebra."""

    name = "base"
    pair: Optional[Dim] = None

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b

    def sub(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a - b

    def neg(self, a: torch.Tensor) -> torch.Tensor:
        return -a

    def conj(self, a: torch.Tensor) -> torch.Tensor:
        return a

    def abs(self, a: torch.Tensor) -> torch.Tensor:
        return a.abs()

    def norm_vec(self, a: torch.Tensor, dim: int, eps: float = 1e-6) -> torch.Tensor:
        mag = torch.linalg.vector_norm(a, dim=dim, keepdim=True).clamp_min(eps)
        return a / mag

    def exp(self, a: torch.Tensor) -> torch.Tensor:
        return torch.exp(a)

    def __repr__(self) -> str:
        return type(self).__name__ + "()"


class Real(ElementPolicy):
    name = "real"


class NativeComplex(ElementPolicy):
    """torch.complex64/128 — conj/abs are the standard torch ops."""

    name = "native_complex"

    def conj(self, a: torch.Tensor) -> torch.Tensor:
        return torch.conj(a)

    def abs(self, a: torch.Tensor) -> torch.Tensor:
        return torch.abs(a)

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b


class SplitComplex(ElementPolicy):
    """v13-compatible split-real ``[..., 2]`` algebra.

    Epsilons default to v13: ``abs`` uses ``+1e-8`` inside the sqrt;
    ``norm_vec`` uses the exact ``cnormalize_vec`` form (per-vector, +1e-8).
    """

    name = "split_complex"

    def __init__(
        self,
        pair: Dim,
        abs_eps: float = 1e-8,
        norm_eps: float = 1e-8,
    ):
        if pair.size not in (None, 2):
            raise ValueError(f"SplitComplex pair dim must have size 2, got {pair}")
        if pair.size is None:
            pair.bind(2)
        self.pair = pair
        self.abs_eps = abs_eps
        self.norm_eps = norm_eps

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # cmul — last axis is the pair
        return torch.stack(
            [
                a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
                a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
            ],
            dim=-1,
        )

    def conj(self, a: torch.Tensor) -> torch.Tensor:
        return torch.stack([a[..., 0], -a[..., 1]], dim=-1)

    def abs(self, a: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(a[..., 0].square() + a[..., 1].square() + self.abs_eps)

    def norm_vec(self, a: torch.Tensor, dim: int, eps: Optional[float] = None) -> torch.Tensor:
        """Per-VECTOR unit norm across the feature dim (v13 ``cnormalize_vec``).

        ``a`` is ``[..., feat, 2]`` and ``dim`` is the feature axis. Result
        keeps both axes; magnitude is ``sqrt(sum_d (re^2+im^2) + eps)``.
        """
        e = self.norm_eps if eps is None else eps
        # Shift so feat is -2, pair is -1 for the v13 formula, then restore.
        if dim < 0:
            dim = a.ndim + dim
        pair_dim = a.ndim - 1
        x = a
        if dim != a.ndim - 2:
            perm = [i for i in range(a.ndim) if i != dim and i != pair_dim] + [dim, pair_dim]
            x = a.permute(perm)
        mag = torch.sqrt((x[..., 0].square() + x[..., 1].square()).sum(-1) + e)
        out = x / mag.unsqueeze(-1).unsqueeze(-1)
        if dim != a.ndim - 2:
            inv = [0] * a.ndim
            for new_i, old_i in enumerate(perm):
                inv[old_i] = new_i
            out = out.permute(inv)
        return out

    def exp(self, a: torch.Tensor) -> torch.Tensor:
        # exp(re + i im) = exp(re) * (cos(im) + i sin(im))
        re, im = a[..., 0], a[..., 1]
        scale = torch.exp(re)
        return torch.stack([scale * torch.cos(im), scale * torch.sin(im)], dim=-1)


# Module-level singletons for the default real policy.
REAL = Real()
