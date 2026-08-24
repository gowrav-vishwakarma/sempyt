"""Dim objects, ProductDim, dims(), registry, string parse."""

import pytest

from sempyt import Dim, DimRegistry, ProductDim, ShapeError, dims
from sempyt.dim import lookup_from, parse_layout


def test_dim_is_object_identity():
    a = Dim("d")
    b = Dim("d")
    assert a is not b
    assert a != b
    assert hash(a) == id(a)


def test_bind_and_disagree():
    d = Dim("d")
    assert d.size is None
    d.bind(64)
    assert d.size == 64
    assert d.bound_at is not None
    with pytest.raises(ShapeError, match="disagrees"):
        d.bind(128)


def test_bind_at_construction():
    d = Dim("d", 64)
    assert d.size == 64
    assert d.bound_at is not None


def test_dims_batch_constructor():
    B, T, H, d = dims("B T H d")
    assert isinstance(B, Dim) and B.name == "B"
    assert T.name == "T" and H.name == "H" and d.name == "d"
    assert B is not T


def test_product_dim_derived_size():
    H, d = Dim("H", 6), Dim("d", 64)
    HD = H * d
    assert isinstance(HD, ProductDim)
    assert HD.size == 384
    assert HD.leaves() == (H, d)


def test_product_nested_and_kernel_flatten():
    K, B, H = Dim("K", 3), Dim("B", 2), Dim("H", 6)
    Kf = K * B * H
    assert Kf.size == 36
    assert Kf.leaves() == (K, B, H)


def test_product_bind_factors_one_unbound():
    H, d = Dim("H", 6), Dim("d")
    HD = H * d
    HD.bind_product(384)
    assert d.size == 64


def test_product_underdetermined():
    H, d = Dim("H"), Dim("d")
    HD = H * d
    with pytest.raises(ShapeError, match="under-determined"):
        HD.bind_product(384)


def test_role_tag():
    px = Dim("px", 2, role="complex_pair")
    assert px.role == "complex_pair"


def test_registry_string_lookup():
    B, T, d = dims("B T d")
    with DimRegistry(B, T, d) as reg:
        assert reg.get("B") is B
        parsed = parse_layout("B d T", lookup_from())
        assert parsed[0] is B and parsed[1] is d and parsed[2] is T


def test_parse_product_group():
    B, H, d = dims("B H d")
    parsed = parse_layout("B (H d)", lookup_from((B, H, d)))
    assert parsed[0] is B
    assert isinstance(parsed[1], ProductDim)
    assert parsed[1].leaves() == (H, d)
