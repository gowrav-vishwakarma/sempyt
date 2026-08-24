"""Layout algebra: permute, split, merge, errors, no-op identity."""

import pytest
import torch

from sempyt import Dim, ProductDim, ShapeError, dims, named
from sempyt.testing import assert_close, assert_same_object


def _btHd():
    B, T, H, d = Dim("B", 2), Dim("T", 4), Dim("H", 3), Dim("d", 8)
    data = torch.arange(2 * 4 * 3 * 8, dtype=torch.float32).reshape(2, 4, 3, 8)
    x = named(data, (B, T, H, d))
    return x, B, T, H, d


def test_shape_rejects_int_index():
    x, B, T, H, d = _btHd()
    assert x.shape[B] == 2
    assert x.shape[d] == 8
    with pytest.raises(TypeError, match="keyed by Dim"):
        _ = x.shape[0]


def test_permute_by_names():
    x, B, T, H, d = _btHd()
    y = x.to(B, H, T, d)
    assert y.layout == (B, H, T, d)
    assert_close(y.data, x.data.permute(0, 2, 1, 3))


def test_noop_to_returns_self():
    x, B, T, H, d = _btHd()
    y = x.to(B, T, H, d)
    assert_same_object(x, y)
    assert_same_object(x.data, y.data)


def test_product_split_and_merge():
    B, T, H, d = Dim("B", 2), Dim("T", 4), Dim("H", 3), Dim("d", 8)
    HD = H * d
    data = torch.randn(2, 4, 24)
    x = named(data, (B, T, HD))
    y = x.to(B, T, H, d)
    assert y.layout == (B, T, H, d)
    assert tuple(y.data.shape) == (2, 4, 3, 8)
    z = y.to(B, T, HD)
    assert_close(z.data, data)
    # merge via a fresh ProductDim of the same factors
    z2 = y.to(B, T, H * d)
    assert_close(z2.data, data)


def test_kernel_flatten_KBH():
    K, B, H, C = Dim("K", 3), Dim("B", 2), Dim("H", 4), Dim("C", 5)
    Kf = K * B * H
    data = torch.randn(3, 2, 4, 5, 5)
    x = named(data, (K, B, H, C, C))
    raw = x.raw(Kf, C, C)
    assert raw.shape == (24, 5, 5)
    assert_close(raw, data.reshape(24, 5, 5))
    back = named(raw, (Kf, C, C)).to(K, B, H, C, C)
    assert_close(back.data, data)


def test_size_factor_split_design_md():
    B, T, D = Dim("B", 2), Dim("T", 4), Dim("D", 24)
    H, d = Dim("H", 3), Dim("d", 8)
    data = torch.randn(2, 4, 24)
    x = named(data, (B, T, D))
    y = x.to(B, T, H, d)
    assert tuple(y.data.shape) == (2, 4, 3, 8)
    z = y.to(B, T, D)
    assert_close(z.data, data)


def test_pack_unpack_split_real():
    B, d, px = Dim("B", 2), Dim("d", 8), Dim("px", 2, role="complex_pair")
    d2 = px * d
    data = torch.randn(2, 8, 2)
    x = named(data, (B, d, px))
    packed = x.to(B, d2)
    assert packed.data.shape == (2, 16)
    # unpack via product (px * d) is (px, d) order — row-major of (px, d)
    # Our merge of (d, px) is d-major then px (last). pack (B, d, px) -> (B, d*px)
    # which is NOT (px * d) if we define px * d as px-outer.
    # to(B, px * d) would want leaves (px, d) but data leaves are (d, px).
    # That's a permute+merge: (B, d, px) -> (B, px, d) -> (B, px*d)
    y = x.to(B, px, d)
    packed2 = y.to(B, d2)
    assert packed2.data.shape == (2, 16)
    back = packed2.to(B, px, d)
    assert_close(back.data, y.data)


def test_string_form_to():
    x, B, T, H, d = _btHd()
    y = x.to("B H T d")
    assert y.layout == (B, H, T, d)
    assert_close(y.data, x.data.permute(0, 2, 1, 3))


def test_string_product_group():
    B, T, H, d = Dim("B", 2), Dim("T", 4), Dim("H", 3), Dim("d", 8)
    HD = H * d
    x = named(torch.randn(2, 4, 24), (B, T, HD))
    y = x.to("B T H d")
    assert y.layout == (B, T, H, d)


def test_insert_and_squeeze_size_one():
    B, d = Dim("B", 2), Dim("d", 4)
    u = Dim("u", 1)
    x = named(torch.randn(2, 4), (B, d))
    y = x.to(B, u, d)
    assert y.data.shape == (2, 1, 4)
    z = y.to(B, d)
    assert_close(z.data, x.data)


def test_mismatch_names_error():
    x, B, T, H, d = _btHd()
    Q = Dim("Q", 7)
    with pytest.raises(ShapeError, match="cannot match"):
        x.to(B, Q, d)


def test_size_disagree_on_bind():
    B, T = Dim("B", 2), Dim("T", 4)
    with pytest.raises(ShapeError, match="disagrees"):
        named(torch.randn(3, 4), (B, T))


def test_alias_for_attention_two_times():
    x, B, T, H, d = _btHd()
    Tq, Tk = Dim("Tq"), Dim("Tk")
    q = x.alias(T, Tq)
    k = x.alias(T, Tk)
    assert Tq.size == 4 and Tk.size == 4
    assert q.data.data_ptr() == x.data.data_ptr()
