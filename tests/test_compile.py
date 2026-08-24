"""torch.compile fullgraph + no-op to() identity + a tiny overhead check."""

import time

import torch

from sempyt import Dim, named
from sempyt.nn import Linear
from sempyt.testing import assert_close, assert_same_object


def _block():
    inn, out = Dim("in", 16), Dim("out", 16)
    B, T, H, d = Dim("B"), Dim("T"), Dim("H", 2), Dim("dh", 8)
    layer = Linear(inn, out, bias=True)

    def fn(x_raw):
        b, t, _ = x_raw.shape
        B.bind(b) if B.size is None else None
        T.bind(t) if T.size is None else None
        x = named(x_raw, (B, T, inn))
        y = layer(x)
        z = y.to(B, T, H, d).to(B, T, inn)
        return z.data

    return fn, layer, inn


def test_noop_to_same_object():
    B, T, d = Dim("B", 2), Dim("T", 4), Dim("d", 8)
    x = named(torch.randn(2, 4, 8), (B, T, d))
    assert_same_object(x.to(B, T, d), x)


def test_compile_linear_rearrange():
    fn, layer, inn = _block()
    x = torch.randn(2, 5, 16)
    eager = fn(x)
    # Rebind dims for the compiled call: Dim sizes stay from eager bind
    compiled = torch.compile(fn, fullgraph=True)
    out = compiled(x)
    assert_close(out, eager, rtol=1e-5, atol=1e-6)


def test_overhead_noop_and_tiny_block():
    B, T, d = Dim("B", 4), Dim("T", 32), Dim("d", 64)
    x = named(torch.randn(4, 32, 64), (B, T, d))
    t0 = time.perf_counter()
    for _ in range(200):
        x.to(B, T, d)
    named_dt = time.perf_counter() - t0
    raw = x.data
    t0 = time.perf_counter()
    for _ in range(200):
        raw.permute(0, 1, 2)
    raw_dt = time.perf_counter() - t0
    # no-op to() must be cheap — not 100x a permute of an already-correct layout
    # (plan cache + identity return)
    assert named_dt < 0.05 or named_dt < raw_dt * 50
