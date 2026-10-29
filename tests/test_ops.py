"""contract, outer, broadcast, reduce, segment, gather, scan."""

import torch

from sempyt import (
    Dim,
    Index,
    ShapeError,
    SplitComplex,
    at,
    contract,
    expand_legs,
    gather,
    named,
    outer,
    pack,
    read,
    scan,
    segment,
    softmax,
    sum,
    unsegment,
    unpack,
    write,
)
from sempyt.testing import assert_close


def test_named_broadcast_deletes_unsqueeze():
    B, H, d_i, d_j = Dim("B", 2), Dim("H", 3), Dim("di", 4), Dim("dj", 4)
    state = named(torch.randn(2, 3, 4, 4), (B, H, d_i, d_j))
    gamma = named(torch.randn(2, 3), (B, H))
    out = state * gamma
    expected = state.data * gamma.data[:, :, None, None]
    assert_close(out, expected)


def test_contract_attention_scores():
    B, Tq, Tk, H, d = Dim("B", 2), Dim("Tq", 5), Dim("Tk", 7), Dim("H", 3), Dim("d", 8)
    q = named(torch.randn(2, 5, 3, 8), (B, Tq, H, d))
    k = named(torch.randn(2, 7, 3, 8), (B, Tk, H, d))
    scores = contract(q, k, over=d)
    # q: B Tq H d, k: B Tk H d → B Tq H Tk
    q_ = q.to(B, H, Tq, d).data
    k_ = k.to(B, H, Tk, d).data
    exp = torch.einsum("bhqd,bhkd->bhqk", q_, k_)
    got = scores.to(B, H, Tq, Tk).data
    assert_close(got, exp)


def test_outer_and_write():
    B, di, dj = Dim("B", 2), Dim("di", 4), Dim("dj", 4)
    u = named(torch.randn(2, 4), (B, di))
    k = named(torch.randn(2, 4), (B, dj))
    S = named(torch.zeros(2, 4, 4), (B, di, dj))
    S2 = write(S, u, k, over=(di, dj), conjugate=False)
    exp = u.data.unsqueeze(-1) * k.data.unsqueeze(-2)
    assert_close(S2.data, exp)


def test_read_is_inner_product():
    B, d = Dim("B", 2), Dim("d", 8)
    S = named(torch.randn(2, 8), (B, d))
    p = named(torch.randn(2, 8), (B, d))
    out = read(S, p, over=d, conjugate="none")
    exp = (S.data * p.data).sum(-1)
    assert_close(out.data, exp)


def test_softmax_over_named():
    B, T = Dim("B", 2), Dim("T", 5)
    x = named(torch.randn(2, 5), (B, T))
    y = softmax(x, over=T)
    assert_close(y.data, torch.softmax(x.data, dim=1))


def test_sum_drops_axis():
    B, T, d = Dim("B", 2), Dim("T", 3), Dim("d", 4)
    x = named(torch.randn(2, 3, 4), (B, T, d))
    y = sum(x, over=T)
    assert y.layout == (B, d)
    assert_close(y.data, x.data.sum(1))


def test_segment_roundtrip():
    B, T, d = Dim("B", 2), Dim("T", 10), Dim("d", 4)
    S, L = Dim("S"), Dim("L", 4)
    x = named(torch.randn(2, 10, 4), (B, T, d))
    y, info = segment(x, T, into=(S, L), pad_value=0.0)
    assert S.size == 3
    assert y.data.shape == (2, 3, 4, 4) or set(y.data.shape) == {2, 3, 4}
    z = unsegment(y, info)
    assert_close(z.data, x.data)
    assert z.size(T) == 10


def test_expand_legs():
    B, S, d = Dim("B", 2), Dim("S", 3), Dim("d", 4)
    T, L = Dim("T"), Dim("L", 4)
    legs = named(torch.randn(2, 3, 4), (B, S, d))
    tok = expand_legs(legs, S, T, L, orig_T=10)
    assert tok.data.shape[1] == 10
    # first 4 tokens of seq == first leg
    assert_close(tok.data[:, :4], legs.data[:, 0:1].expand(2, 4, 4)[:, :4] * 0 + legs.data[:, 0, :].unsqueeze(1).expand(2, 4, 4))


def test_gather_table():
    V, d = Dim("V", 10), Dim("d", 4)
    B, T = Dim("B", 2), Dim("T", 3)
    table = named(torch.randn(10, 4), (V, d))
    ids = torch.randint(0, 10, (2, 3))
    idx = Index(ids, (B, T), over=V)
    out = at(table, idx, over=V)
    assert_close(out.data, table.data[ids])


def test_scan_three_modes_agree_real():
    B, T, d = Dim("B", 2), Dim("T", 12), Dim("d", 4)
    u = named(torch.randn(2, 12, 4), (B, T, d))
    lam = named(torch.sigmoid(torch.randn(4)) * 0.9 + 0.05, (d,))
    rec = scan(u, lam, over=T, mode="recurrent")
    par = scan(u, lam, over=T, mode="parallel")
    chk = scan(u, lam, over=T, mode="chunked", chunk=5)
    assert_close(par, rec, rtol=1e-4, atol=1e-5)
    assert_close(chk, rec, rtol=1e-4, atol=1e-5)


def test_pack_unpack():
    B, H, d = Dim("B", 2), Dim("H", 3), Dim("d", 8)
    x = named(torch.randn(2, 3, 8), (B, H, d))
    y = pack(x, (H, d))
    assert y.data.shape == (2, 24)
    z = unpack(y, y.layout[-1], (H, d))
    assert_close(z.data, x.data)


def test_scan_per_step_decay_is_exact_at_long_length():
    """Per-step γ over 600 steps (Π γ ≈ 1e-28): chunked/parallel == recurrent,
    with the decay broadcast over channels and a split-complex input."""
    import torch
    from sempyt import Dim, SplitComplex, named, scan
    torch.manual_seed(0)
    B, C, T, pair = Dim("B", 2), Dim("C", 3), Dim("T", 600), Dim("pair", 2)
    g = named(0.85 + 0.1 * torch.rand(2, 600), (B, T))
    u = named(torch.randn(2, 3, 600), (B, C, T))
    rec = scan(u, g, over=T, mode="recurrent").to(B, C, T).data
    for mode in ("parallel", "chunked"):
        got = scan(u, g, over=T, mode=mode).to(B, C, T).data
        assert torch.allclose(got, rec, atol=1e-4), (mode, (got - rec).abs().max())
    z = named(torch.randn(2, 3, 2, 600), (B, C, pair, T), policy=SplitComplex(pair))
    rec = scan(z, g, over=T, mode="recurrent").to(B, C, pair, T).data
    got = scan(z, g, over=T, mode="chunked").to(B, C, pair, T).data
    assert torch.allclose(got, rec, atol=1e-4), (got - rec).abs().max()


def test_scan_per_step_decay_gradients_match_recurrent():
    import torch
    from sempyt import Dim, named, scan
    torch.manual_seed(1)
    B, C, T = Dim("B", 2), Dim("C", 4), Dim("T", 150)
    g0 = (0.5 + 0.45 * torch.rand(2, 150)).requires_grad_(True)
    u0 = torch.randn(2, 4, 150, requires_grad=True)
    w = torch.randn(2, 4, 150)
    grads = []
    for mode in ("recurrent", "chunked"):
        out = scan(named(u0, (B, C, T)), named(g0, (B, T)), over=T, mode=mode).to(B, C, T).data
        gu, gg = torch.autograd.grad((out * w).sum(), (u0, g0))
        grads.append((gu, gg))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-4)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-3)
