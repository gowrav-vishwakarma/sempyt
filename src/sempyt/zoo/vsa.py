"""Vector Symbolic Architecture (phasor HDC) with named axes.

Hypervectors are split-complex phasors on a named hypervector axis ``hd`` with
the pair axis ``px``. bind = complex multiply, unbind = multiply by conjugate,
bundle = superposition (sum over an ``items`` axis), cleanup = nearest codebook
entry by real similarity. Everything addresses axes by name.
"""

from __future__ import annotations

import torch

from ..dim import Dim
from ..ops import Index, at, read
from ..policies import SplitComplex
from ..structural import argmax, select, stack, zeros_like
from ..tensor import NamedTensor, named


def random_phasors(*shape_dims: Dim, hd: Dim, px: Dim, seed: int | None = None) -> NamedTensor:
    """Random unit phasors: layout ``(*shape_dims, hd, px)``."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    sizes = [d.size for d in shape_dims] + [hd.size]
    theta = torch.rand(*sizes, generator=g) * (2 * torch.pi)
    data = torch.stack([theta.cos(), theta.sin()], dim=-1)
    return named(data, tuple(shape_dims) + (hd, px), SplitComplex(px))


def bind(a: NamedTensor, b: NamedTensor) -> NamedTensor:
    """Binding = elementwise complex multiply (rotate phases)."""
    return a * b


def unbind(bound: NamedTensor, key: NamedTensor) -> NamedTensor:
    """Unbind by multiplying with the conjugate of the key."""
    return bound * key.conj()


def bundle(items: list[NamedTensor], into: Dim) -> NamedTensor:
    """Superpose a list of hypervectors (stack over ``into`` then sum it)."""
    from ..ops import sum as nsum

    stacked = stack(items, into=into)
    return nsum(stacked, over=into)


def cleanup(query: NamedTensor, codebook: NamedTensor, code: Dim, hd: Dim) -> tuple[Index, NamedTensor]:
    """Nearest codebook entry to ``query`` by real similarity over ``hd``.

    Returns (winning Index over ``code``, the cleaned-up hypervector).
    """
    sim_c = read(codebook, query, over=hd)      # complex similarity, layout has `code`
    sim = select(sim_c, sim_c.policy.pair, 0)    # real part
    idx = argmax(sim, over=code)
    return idx, at(codebook, idx, over=code)
