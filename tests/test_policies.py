"""Element policies: Real, SplitComplex (v13-matching), NativeComplex."""

import torch

from sempyt import Dim, NativeComplex, SplitComplex, as_complex, imag, named, real
from sempyt.testing import assert_close


def test_split_complex_cmul_matches_v13():
    px = Dim("px", 2, role="complex_pair")
    B, d = Dim("B", 3), Dim("d", 5)
    pol = SplitComplex(px)
    a = named(torch.randn(3, 5, 2), (B, d, px), pol)
    b = named(torch.randn(3, 5, 2), (B, d, px), pol)
    out = a * b
    exp = torch.stack(
        [
            a.data[..., 0] * b.data[..., 0] - a.data[..., 1] * b.data[..., 1],
            a.data[..., 0] * b.data[..., 1] + a.data[..., 1] * b.data[..., 0],
        ],
        dim=-1,
    )
    assert_close(out.data, exp)


def test_split_complex_conj_abs():
    px = Dim("px", 2, role="complex_pair")
    d = Dim("d", 4)
    pol = SplitComplex(px)
    z = named(torch.randn(4, 2), (d, px), pol)
    c = z.conj()
    assert_close(c.data[..., 0], z.data[..., 0])
    assert_close(c.data[..., 1], -z.data[..., 1])
    mag = z.abs()
    exp = torch.sqrt(z.data[..., 0].square() + z.data[..., 1].square() + 1e-8)
    assert_close(mag.data, exp)


def test_cnormalize_vec_v13_form():
    px = Dim("px", 2, role="complex_pair")
    B, H, T, d = Dim("B", 2), Dim("H", 3), Dim("T", 4), Dim("d", 8)
    pol = SplitComplex(px)
    x = named(torch.randn(2, 3, 4, 8, 2), (B, H, T, d, px), pol)
    y = x.normalize(over=d)
    mag = torch.sqrt((x.data[..., 0].square() + x.data[..., 1].square()).sum(-1) + 1e-8)
    exp = x.data / mag.unsqueeze(-1).unsqueeze(-1)
    assert_close(y.data, exp)


def test_native_complex_uses_torch_conj():
    B, d = Dim("B", 2), Dim("d", 4)
    z = named(torch.randn(2, 4, dtype=torch.complex64), (B, d), NativeComplex())
    c = z.conj()
    assert_close(c.data, torch.conj(z.data))


def test_real_times_complex_broadcasts():
    px = Dim("px", 2, role="complex_pair")
    B, d = Dim("B", 2), Dim("d", 4)
    pol = SplitComplex(px)
    z = named(torch.randn(2, 4, 2), (B, d, px), pol)
    scale = named(torch.randn(2, 4), (B, d))
    out = z * scale
    assert_close(out.data, z.data * scale.data.unsqueeze(-1))


def test_real_imag_as_complex_roundtrip():
    px = Dim("px", 2, role="complex_pair")
    B, d = Dim("B", 2), Dim("d", 4)
    z = named(torch.randn(2, 4, 2), (B, d, px), SplitComplex(px))
    packed = as_complex(real(z), imag(z), px)
    assert_close(packed.data, z.data)
    assert packed.policy.pair is px


def test_split_complex_ops_refuse_pair_not_last():
    """abs / conj / normalize read (re, im) from the last axis: if the pair
    axis is elsewhere they must raise, not silently mix numbers."""
    import pytest
    import torch
    from sempyt import Dim, ShapeError, SplitComplex, named
    T, F, pair = Dim("T", 5), Dim("F", 3), Dim("pair", 2)
    z_last = named(torch.randn(5, 3, 2), (T, F, pair), policy=SplitComplex(pair))
    expect = torch.sqrt((z_last.data ** 2).sum(-1))
    assert torch.allclose(z_last.abs().data, expect)
    z_moved = z_last.to(T, pair, F)                    # pair no longer last
    for op in (lambda z: z.abs(), lambda z: z.conj(), lambda z: z.normalize(F)):
        with pytest.raises(ShapeError, match="must be last"):
            op(z_moved)
    assert torch.allclose(z_moved.to(T, F, pair).abs().data, expect)
