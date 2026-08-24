"""Named structural + elementwise ops — no numeric dim= anywhere in model code."""

import torch

import sempyt as sp
from sempyt import Dim, dims, named


def _rand(*ds):
    return named(torch.randn(*[d.size for d in ds]), ds)


def test_stack_unstack_roundtrip():
    B, T, D = dims("batch time dim", batch=2, time=3, dim=4)
    heads = Dim("heads")
    xs = [_rand(B, T, D) for _ in range(5)]
    y = sp.stack(xs, into=heads)
    assert y.layout[-1] is heads and heads.size == 5
    assert y.data.shape == (2, 3, 4, 5)
    back = sp.unstack(y, heads)
    assert len(back) == 5
    torch.testing.assert_close(back[2].data, xs[2].data)


def test_stack_at_position():
    B, T, D = dims("batch time dim", batch=2, time=3, dim=4)
    heads = Dim("heads")
    y = sp.stack([_rand(B, T, D) for _ in range(6)], into=heads, at=T)
    assert y.layout == (B, heads, T, D)


def test_cat_over_named():
    B, T, D = dims("batch time dim", batch=2, time=3, dim=4)
    a = named(torch.randn(2, 3, 4), (B, T, D))
    b = named(torch.randn(2, 3, 4), (B, T, D))
    c = sp.cat([a, b], over=T)
    assert c.data.shape == (2, 6, 4)
    torch.testing.assert_close(c.data, torch.cat([a.data, b.data], dim=1))


def test_split_named():
    B, D = dims("batch dim", batch=2, dim=12)
    x = _rand(B, D)
    q, k, v = dims("q k v", q=4, k=4, v=4)
    a, b, c = sp.split(x, over=D, sizes=(4, 4, 4), names=(q, k, v))
    assert a.layout == (B, q) and c.layout == (B, v)
    torch.testing.assert_close(a.data, x.data[:, :4])
    torch.testing.assert_close(c.data, x.data[:, 8:])


def test_select_and_getitem():
    B, T, D = dims("batch time dim", batch=2, time=5, dim=4)
    x = _rand(B, T, D)
    last = sp.select(x, T, -1)
    assert last.layout == (B, D)
    torch.testing.assert_close(last.data, x.data[:, -1])
    torch.testing.assert_close(x[T, 0].data, x.data[:, 0])


def test_take_narrow():
    B, T, D = dims("batch time dim", batch=2, time=8, dim=4)
    x = _rand(B, T, D)
    win = Dim("window")
    y = sp.take(x, T, start=2, length=3, new=win)
    assert y.layout == (B, win, D) and win.size == 3
    torch.testing.assert_close(y.data, x.data[:, 2:5])


def test_cumsum_cumprod():
    B, T = dims("batch time", batch=2, time=4)
    x = _rand(B, T)
    torch.testing.assert_close(sp.cumsum(x, T).data, x.data.cumsum(1))
    torch.testing.assert_close(x.cumprod(T).data, x.data.cumprod(1))


def test_argmax_topk():
    B, V = dims("batch vocab", batch=3, vocab=10)
    logits = _rand(B, V)
    idx = sp.argmax(logits, V)
    assert idx.values.shape == (3,)
    torch.testing.assert_close(idx.values, logits.data.argmax(1))
    k = Dim("k")
    vals, ids = sp.topk(logits, V, k=4, kdim=k)
    assert vals.layout == (B, k) and k.size == 4
    torch.testing.assert_close(vals.data, logits.data.topk(4, dim=1).values)


def test_repeat_along_gqa():
    B, KV, T, Dh = dims("batch kvheads time headdim", batch=2, kvheads=2, time=3, headdim=4)
    kv = _rand(B, KV, T, Dh)
    H = Dim("heads")
    q = sp.repeat_along(kv, KV, times=3, new=H)
    assert q.layout == (B, H, T, Dh) and H.size == 6
    torch.testing.assert_close(q.data, kv.data.repeat_interleave(3, dim=1))


def test_flip_roll_pad():
    B, T = dims("batch time", batch=2, time=4)
    x = _rand(B, T)
    torch.testing.assert_close(x.flip(T).data, x.data.flip(1))
    torch.testing.assert_close(x.roll(T, 1).data, x.data.roll(1, dims=1))
    padded = sp.pad(x, T, left=1, right=2)
    assert padded.data.shape == (2, 7)


def test_where_broadcast():
    B, T = dims("batch time", batch=2, time=4)
    x = _rand(B, T)
    cond = named(x.data > 0, (B, T))
    y = sp.where(cond, x, sp.zeros_like(x))
    torch.testing.assert_close(y.data, torch.where(x.data > 0, x.data, torch.zeros_like(x.data)))


def test_unary_maps():
    B, T = dims("batch time", batch=2, time=4)
    x = _rand(B, T)
    torch.testing.assert_close(sp.sigmoid(x).data, x.data.sigmoid())
    torch.testing.assert_close(sp.silu(x).data, torch.nn.functional.silu(x.data))
    torch.testing.assert_close(sp.exp(x).data, x.data.exp())
    assert sp.relu(x).layout == (B, T)
