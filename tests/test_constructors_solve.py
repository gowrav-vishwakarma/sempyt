"""Named constructors (zeros/ones/full/eye/tril_mask) and solve_triangular."""

import pytest
import torch

import sempyt as sp
from sempyt import Dim, ShapeError, SplitComplex, dims, named


def test_zeros_ones_full_from_named_axes():
    batch, heads, time = dims("batch heads time", batch=2, heads=3, time=4)
    z = sp.zeros(batch, heads, time)
    o = sp.ones(batch, heads, time)
    f = sp.full(2.5, batch, heads, time)
    assert z.layout == (batch, heads, time)
    assert z.data.shape == (2, 3, 4)
    assert float(z.data.sum()) == 0.0
    assert float(o.data.sum()) == 24.0
    assert float(f.data.mean()) == pytest.approx(2.5)


def test_zeros_with_split_complex_policy():
    batch, feature = dims("batch feature", batch=2, feature=5)
    pair = Dim("complex_pair", 2)
    z = sp.zeros(batch, feature, pair, policy=SplitComplex(pair))
    assert isinstance(z.policy, SplitComplex)
    assert sp.real(z).layout == (batch, feature)


def test_zeros_rejects_unbound_dim():
    batch = Dim("batch", 2)
    loose = Dim("loose")
    with pytest.raises(ShapeError):
        sp.zeros(batch, loose)


def test_eye_and_tril_mask():
    row, col = dims("row col", row=4, col=4)
    identity = sp.eye(row, col)
    torch.testing.assert_close(identity.data, torch.eye(4))

    inclusive = sp.tril_mask(row, col)
    strict = sp.tril_mask(row, col, diagonal=-1)
    torch.testing.assert_close(inclusive.data, torch.tril(torch.ones(4, 4)))
    torch.testing.assert_close(strict.data, torch.tril(torch.ones(4, 4), -1))
    assert inclusive.layout == (row, col)


def test_eye_rejects_mismatched_axes():
    row, col = dims("row col", row=4, col=5)
    with pytest.raises(ShapeError):
        sp.eye(row, col)


def test_solve_triangular_matches_torch():
    row, col, feature = dims("row col feature", row=5, col=5, feature=3)
    mass = torch.tril(torch.randn(5, 5), -1)
    rhs = torch.randn(5, 3)
    system = named(mass, (row, col))
    x = sp.solve_triangular(
        system, named(rhs, (row, feature)),
        over=(row, col), feature=feature, unitriangular=True,
    )
    assert x.layout == (col, feature)
    expected = torch.linalg.solve_triangular(
        mass, rhs, upper=False, unitriangular=True
    )
    torch.testing.assert_close(x.data, expected)


def test_solve_triangular_broadcasts_rhs_only_axis():
    """One system, a state-batched rhs: no packing, same answer per state."""
    states, row, col, feature = dims(
        "states row col feature", states=3, row=4, col=4, feature=2
    )
    mass = torch.tril(torch.randn(4, 4), -1)
    rhs = torch.randn(3, 4, 2)
    x = sp.solve_triangular(
        named(mass, (row, col)), named(rhs, (states, row, feature)),
        over=(row, col), feature=feature, unitriangular=True,
    )
    assert x.layout == (states, col, feature)
    for k in range(3):
        expected = torch.linalg.solve_triangular(
            mass, rhs[k], upper=False, unitriangular=True
        )
        torch.testing.assert_close(x.data[k], expected)


def test_solve_triangular_complex_split_policy():
    row, col, feature = dims("row col feature", row=4, col=4, feature=2)
    pair = Dim("complex_pair", 2)
    policy = SplitComplex(pair)
    mass_r = torch.tril(torch.randn(4, 4), -1)
    mass_i = torch.tril(torch.randn(4, 4), -1)
    rhs_r, rhs_i = torch.randn(4, 2), torch.randn(4, 2)
    system = named(torch.stack([mass_r, mass_i], dim=-1), (row, col, pair), policy)
    rhs = named(torch.stack([rhs_r, rhs_i], dim=-1), (row, feature, pair), policy)

    x = sp.solve_triangular(
        system, rhs, over=(row, col), feature=feature, unitriangular=True
    )
    expected = torch.linalg.solve_triangular(
        torch.complex(mass_r, mass_i), torch.complex(rhs_r, rhs_i),
        upper=False, unitriangular=True,
    )
    torch.testing.assert_close(sp.real(x).data, expected.real)
    torch.testing.assert_close(sp.imag(x).data, expected.imag)


def test_solve_triangular_ignores_unit_diagonal():
    """unitriangular=True never reads the diagonal, so I+M == M."""
    row, col, feature = dims("row col feature", row=4, col=4, feature=2)
    mass = torch.tril(torch.randn(4, 4), -1)
    rhs = named(torch.randn(4, 2), (row, feature))
    bare = sp.solve_triangular(
        named(mass, (row, col)), rhs,
        over=(row, col), feature=feature, unitriangular=True,
    )
    with_identity = sp.solve_triangular(
        named(mass + torch.eye(4), (row, col)), rhs,
        over=(row, col), feature=feature, unitriangular=True,
    )
    torch.testing.assert_close(bare.data, with_identity.data)


def test_arange_eq_ne_build_named_masks():
    states = Dim("states", 3)
    positions = sp.arange(states)
    assert positions.layout == (states,)
    torch.testing.assert_close(positions.data, torch.arange(3))
    torch.testing.assert_close(sp.eq(positions, 1).data, torch.tensor([False, True, False]))
    torch.testing.assert_close(sp.ne(positions, 1).data, torch.tensor([True, False, True]))


def test_flatten_merges_adjacent_axes_into_one():
    batch, time, feature = dims("batch time feature", batch=2, time=3, feature=4)
    x = named(torch.randn(2, 3, 4), (batch, time, feature))
    flat = Dim("flat")
    y = sp.flatten(x, (batch, time), flat)
    assert y.layout == (flat, feature)
    assert y.data.shape == (6, 4)
    torch.testing.assert_close(y.data, x.data.reshape(6, 4))
    assert flat.size == 6


def test_flatten_reorders_non_adjacent_axes():
    batch, time, feature = dims("batch time feature", batch=2, time=3, feature=4)
    x = named(torch.randn(3, 2, 4), (time, batch, feature))
    flat = Dim("flat", 6)
    y = sp.flatten(x, (batch, time), flat)
    assert y.layout == (flat, feature)
    torch.testing.assert_close(y.data, x.to(batch, time, feature).data.reshape(6, 4))


def test_flatten_rejects_size_mismatch():
    batch, time, feature = dims("batch time feature", batch=2, time=3, feature=4)
    x = named(torch.randn(2, 3, 4), (batch, time, feature))
    flat = Dim("flat", 7)
    with pytest.raises(ShapeError):
        sp.flatten(x, (batch, time), flat)


def test_unflatten_inverts_flatten():
    batch, time, feature = dims("batch time feature", batch=2, time=3, feature=4)
    x = named(torch.randn(2, 3, 4), (batch, time, feature))
    flat = Dim("flat")
    y = sp.flatten(x, (batch, time), flat)
    back = sp.unflatten(y, flat, (batch, time))
    assert back.layout == (batch, time, feature)
    torch.testing.assert_close(back.data, x.data)


def test_unflatten_rejects_bad_product():
    batch, time = dims("batch time", batch=2, time=3)
    flat = Dim("flat", 6)
    y = named(torch.randn(6), (flat,))
    with pytest.raises(ShapeError):
        sp.unflatten(y, flat, (batch, Dim("time", 4)))


def test_cat_matches_concat_axis_by_name_with_ragged_pieces():
    """A chunked loop whose last chunk is short still concatenates."""
    batch, feature = dims("batch feature", batch=2, feature=3)
    first = named(torch.randn(2, 4, 3), (batch, Dim("chunk", 4), feature))
    last = named(torch.randn(2, 1, 3), (batch, Dim("chunk", 1), feature))
    joined = sp.cat([first, last], over="chunk")
    assert joined.data.shape == (2, 5, 3)
    assert joined.layout[1].name == "chunk" and joined.layout[1].size == 5
    torch.testing.assert_close(
        joined.data, torch.cat([first.data, last.data], dim=1)
    )


def test_cat_into_names_the_joined_axis():
    batch, time = dims("batch time", batch=2, time=5)
    pieces = [
        named(torch.randn(2, 4), (batch, Dim("chunk", 4))),
        named(torch.randn(2, 1), (batch, Dim("chunk", 1))),
    ]
    joined = sp.cat(pieces, over="chunk", into=time)
    assert joined.layout == (batch, time)


def test_cat_into_rejects_wrong_size():
    batch, time = dims("batch time", batch=2, time=9)
    pieces = [named(torch.randn(2, 4), (batch, Dim("chunk", 4)))]
    with pytest.raises(ShapeError):
        sp.cat(pieces, over="chunk", into=time)


def test_cat_reorders_other_axes_to_the_first_piece():
    batch, feature = dims("batch feature", batch=2, feature=3)
    chunk_a, chunk_b = Dim("chunk", 4), Dim("chunk", 2)
    first = named(torch.randn(2, 4, 3), (batch, chunk_a, feature))
    second = named(torch.randn(3, 2, 2), (feature, chunk_b, batch))
    joined = sp.cat([first, second], over="chunk")
    assert joined.layout[0] is batch and joined.layout[2] is feature
    torch.testing.assert_close(
        joined.data,
        torch.cat([first.data, second.to(batch, chunk_b, feature).data], dim=1),
    )


def test_solve_triangular_requires_named_axes():
    row, col, feature, other = dims(
        "row col feature other", row=3, col=3, feature=2, other=2
    )
    system = named(torch.tril(torch.randn(3, 3), -1), (row, col))
    with pytest.raises(ShapeError):
        sp.solve_triangular(
            system, named(torch.randn(3, 2), (other, feature)),
            over=(row, col), feature=feature,
        )
