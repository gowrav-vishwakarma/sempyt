"""Semantic ops: contract, outer, reduce, segment, gather, scan, pack/unpack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from .dim import Dim, ProductDim
from .errors import ShapeError, missing
from .layout import layout_str, leaves_of
from .policies import REAL, SplitComplex
from .tensor import NamedTensor, named

Over = Union[Dim, Sequence[Dim]]


def _overs(over: Over) -> tuple[Dim, ...]:
    if isinstance(over, Dim):
        return (over,)
    return tuple(over)


def _letters(n: int) -> list[str]:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if n <= len(alphabet):
        return list(alphabet[:n])
    return [f"a{i}" for i in range(n)]  # torch.einsum rejects multi-char; cap instead


def _einsum_letter(i: int) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if i >= len(alphabet):
        raise ShapeError(f"einsum needs at most {len(alphabet)} distinct axes")
    return alphabet[i]


def _einsum_subs(*layouts: Sequence[Dim], drop: Sequence[Dim] = ()) -> tuple[list[str], str]:
    """Assign one letter per Dim identity across layouts. ``drop`` omitted from output."""
    mapping: dict[int, str] = {}
    i = 0
    specs = []
    for layout in layouts:
        spec = []
        for d in layout:
            k = id(d)
            if k not in mapping:
                mapping[k] = _einsum_letter(i)
                i += 1
            spec.append(mapping[k])
        specs.append("".join(spec))
    drop_ids = {id(d) for d in drop}
    # output = first layout's leftover + other-layouts' unique leftover, stable
    out = []
    seen = set()
    for layout in layouts:
        for d in layout:
            if id(d) in drop_ids:
                continue
            if id(d) in seen:
                continue
            seen.add(id(d))
            out.append(mapping[id(d)])
    return specs, "".join(out)


# ---------------------------------------------------------------------------
# contract / outer / read / write
# ---------------------------------------------------------------------------
def contract(x: NamedTensor, y: NamedTensor, over: Over) -> NamedTensor:
    """Named matmul / inner product: sum over shared axis ``over``.

    Dims in both that are not contracted are batch (kept once). Dims only in
    ``x`` or only in ``y`` are kept. SplitComplex contracts with the 4-real
    GEMM form and keeps the pair axis.
    """
    overs = _overs(over)
    for d in overs:
        if not x.has(d):
            raise missing("contract", d, "x", x.layout)
        if not y.has(d):
            raise missing("contract", d, "y", y.layout)
        dx = x.layout[x.index(d)]
        dy = y.layout[y.index(d)]
        if dx.size is not None and dy.size is not None and dx.size != dy.size:
            from .errors import disagree

            raise disagree(dx, dx.size, dy.size, other=dy)

    both_split = isinstance(x.policy, SplitComplex) and isinstance(y.policy, SplitComplex)
    pair = x.policy.pair if both_split else None
    if pair is not None and any(d is pair for d in overs):
        raise ShapeError("contract: refuse to contract the complex-pair axis; use Real policy")

    if both_split and pair is not None and x.has(pair) and y.has(pair):
        return _contract_split_complex(x, y, overs, pair)

    specs, out_spec = _einsum_subs(x.layout, y.layout, drop=overs)
    expr = f"{specs[0]},{specs[1]}->{out_spec}"
    data = torch.einsum(expr, x.data, y.data)
    out_layout = _result_layout(x.layout, y.layout, overs)
    return NamedTensor(data, out_layout, x.policy, bind=False)


def _result_layout(xl, yl, overs) -> tuple[Dim, ...]:
    drop = {id(d) for d in overs}
    out = []
    seen = set()
    for d in list(xl) + list(yl):
        if id(d) in drop or id(d) in seen:
            continue
        seen.add(id(d))
        out.append(d)
    return tuple(out)


def _contract_split_complex(
    x: NamedTensor, y: NamedTensor, overs: tuple[Dim, ...], pair: Dim
) -> NamedTensor:
    """Complex contract: C = A @ B on the feature axis, pair kept last.

    Inner (no unique-to-y feature besides overs/pair): sum of cmul.
    Matmul: 4-real GEMM when ``y`` has exactly one extra non-batch axis.
    """
    overs_set = {id(d) for d in overs}
    x_rest = [d for d in x.layout if id(d) not in overs_set and d is not pair]
    y_rest = [d for d in y.layout if id(d) not in overs_set and d is not pair]
    batch = [d for d in x_rest if any(d is e for e in y_rest)]
    left = [d for d in x_rest if not any(d is e for e in batch)]
    right = [d for d in y_rest if not any(d is e for e in batch)]

    xa = x.to(*batch, *left, *overs, pair)
    ya = y.to(*batch, *right, *overs, pair)
    xr, xi = xa.data[..., 0], xa.data[..., 1]
    yr, yi = ya.data[..., 0], ya.data[..., 1]

    # Flatten overs into one axis for matmul
    n_over = 1
    for d in overs:
        n_over *= d.size  # type: ignore[operator]
    b = xr.shape[: len(batch)]
    l = xr.shape[len(batch) : len(batch) + len(left)]
    r = yr.shape[len(batch) : len(batch) + len(right)]

    xr_ = xr.reshape(*b, int(torch.tensor(l).prod().item()) if l else 1, n_over) if True else xr
    # safer reshape
    left_n = 1
    for s in l:
        left_n *= s
    right_n = 1
    for s in r:
        right_n *= s
    batch_n = 1
    for s in b:
        batch_n *= s

    xr_m = xr.reshape(batch_n, left_n, n_over)
    xi_m = xi.reshape(batch_n, left_n, n_over)
    yr_m = yr.reshape(batch_n, right_n, n_over).transpose(-1, -2)  # [B, n, right]
    yi_m = yi.reshape(batch_n, right_n, n_over).transpose(-1, -2)

    # (xr + i xi) @ (yr + i yi)  with y as [n, right]
    out_r = xr_m @ yr_m - xi_m @ yi_m
    out_i = xr_m @ yi_m + xi_m @ yr_m
    out = torch.stack([out_r, out_i], dim=-1)
    out = out.reshape(*b, *l, *r, 2)
    out_layout = tuple(batch + left + right + [pair])
    return NamedTensor(out, out_layout, x.policy, bind=False)


def outer(x: NamedTensor, y: NamedTensor, over: tuple[Dim, Dim]) -> NamedTensor:
    """Named outer product. ``over = (A in x, B in y)`` → result has A then B."""
    A, B = over
    if not x.has(A):
        raise missing("outer", A, "x", x.layout)
    if not y.has(B):
        raise missing("outer", B, "y", y.layout)
    xa = x.to(*[d for d in x.layout if d is not A], A)
    ya = y.to(*[d for d in y.layout if d is not B], B)
    both_split = isinstance(x.policy, SplitComplex) and isinstance(y.policy, SplitComplex)
    if both_split and x.policy.pair is not None:
        pair = x.policy.pair
        # move pair last on both, outer over A,B, cmul
        xa = x.to(*[d for d in x.layout if d is not A and d is not pair], A, pair)
        ya = y.to(*[d for d in y.layout if d is not B and d is not pair], B, pair)
        xd = xa.data.unsqueeze(-2)  # [..., A, 1, px]
        yd = ya.data.unsqueeze(-3)  # [..., 1, B, px]
        out = x.policy.mul(xd, yd)
        # align batch of xa, ya — assume they share leading batch
        out_layout = tuple(d for d in xa.layout if d is not pair) + (B, pair)
        return NamedTensor(out, out_layout, x.policy, bind=False)
    xd = xa.data.unsqueeze(-1)
    yd = ya.data.unsqueeze(-2)
    out = xd * yd
    return NamedTensor(out, xa.layout + (B,), x.policy, bind=False)


def read(S: NamedTensor, probe: NamedTensor, over: Dim, conjugate: str = "probe") -> NamedTensor:
    """Inner product over ``over`` (read/collapse). ``conjugate`` in {probe, none}."""
    if conjugate == "probe":
        probe = probe.conj()
    elif conjugate != "none":
        raise ShapeError(f"read: bad conjugate flag {conjugate!r}")
    return contract(S, probe, over)


def write(
    S: NamedTensor,
    u: NamedTensor,
    k: NamedTensor,
    over: tuple[Dim, Dim],
    conjugate: bool = True,
) -> NamedTensor:
    """``S + outer(u, conj(k))`` over ``over=(A, B)``."""
    kk = k.conj() if conjugate else k
    return S + outer(u, kk, over)


def real(z: NamedTensor) -> NamedTensor:
    """Real part of a SplitComplex tensor. Drops the pair axis; policy becomes Real."""
    pair = getattr(z.policy, "pair", None)
    if pair is None or not z.has(pair):
        raise ShapeError("real(): expected a SplitComplex tensor with a pair axis")
    zl = z.to(*[d for d in z.layout if d is not pair], pair)
    layout = tuple(d for d in zl.layout if d is not pair)
    return NamedTensor(zl.data[..., 0], layout, REAL, bind=False)


def imag(z: NamedTensor) -> NamedTensor:
    """Imaginary part of a SplitComplex tensor. Drops the pair axis; policy becomes Real."""
    pair = getattr(z.policy, "pair", None)
    if pair is None or not z.has(pair):
        raise ShapeError("imag(): expected a SplitComplex tensor with a pair axis")
    zl = z.to(*[d for d in z.layout if d is not pair], pair)
    layout = tuple(d for d in zl.layout if d is not pair)
    return NamedTensor(zl.data[..., 1], layout, REAL, bind=False)


def as_complex(re: NamedTensor, im: NamedTensor, pair: Dim) -> NamedTensor:
    """Pack two real tensors of the same layout into SplitComplex ``[..., pair]``."""
    if [id(d) for d in re.layout] != [id(d) for d in im.layout]:
        raise ShapeError("as_complex: real/imag layouts must be identical")
    if pair.size not in (None, 2):
        raise ShapeError(f"as_complex: pair dim must have size 2, got {pair}")
    if pair.size is None:
        pair.bind(2)
    data = torch.stack([re.data, im.data], dim=-1)
    return NamedTensor(data, tuple(re.layout) + (pair,), SplitComplex(pair), bind=False)


# ---------------------------------------------------------------------------
# reductions / softmax
# ---------------------------------------------------------------------------
def reduce(t: NamedTensor, over: Over, how: str = "sum") -> NamedTensor:
    overs = _overs(over)
    dims = [t.index(d) for d in overs]
    fn = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": lambda x, dim: torch.amax(x, dim=dim),
        "min": lambda x, dim: torch.amin(x, dim=dim),
    }.get(how)
    if fn is None:
        raise ShapeError(f"reduce: unknown how={how!r}")
    data = fn(t.data, dim=dims)
    layout = tuple(d for d in t.layout if not any(d is e for e in overs))
    return NamedTensor(data, layout, t.policy, bind=False)


def sum(t: NamedTensor, over: Over) -> NamedTensor:  # noqa: A001
    return reduce(t, over, "sum")


def mean(t: NamedTensor, over: Over) -> NamedTensor:
    return reduce(t, over, "mean")


def softmax(t: NamedTensor, over: Dim) -> NamedTensor:
    data = torch.softmax(t.data, dim=t.index(over))
    return NamedTensor(data, t.layout, t.policy, bind=False)


def log_softmax(t: NamedTensor, over: Dim) -> NamedTensor:
    data = F.log_softmax(t.data, dim=t.index(over))
    return NamedTensor(data, t.layout, t.policy, bind=False)


# ---------------------------------------------------------------------------
# pack / unpack  (ProductDim merge / split)
# ---------------------------------------------------------------------------
def pack(t: NamedTensor, dims: Sequence[Dim], into: Optional[Dim] = None) -> NamedTensor:
    """Merge ``dims`` (in that order) into one axis. ``into`` defaults to their ProductDim."""
    dims = tuple(dims)
    dest = into if into is not None else ProductDim(*dims)
    rest = [d for d in t.layout if not any(d is e for e in dims)]
    return t.to(*rest, dest)


def unpack(t: NamedTensor, dim: Dim, into: Sequence[Dim]) -> NamedTensor:
    """Split ``dim`` into ``into`` (ProductDim or size-factorization)."""
    rest = [d for d in t.layout if d is not dim]
    # If dim is already a ProductDim of `into`, to() expands it.
    # Otherwise size-factorization applies.
    target = []
    inserted = False
    for d in t.layout:
        if d is dim:
            target.extend(into)
            inserted = True
        else:
            target.append(d)
    if not inserted:
        raise missing("unpack", dim, "tensor", t.layout)
    return t.to(*target)


# ---------------------------------------------------------------------------
# segment / unsegment / expand_to
# ---------------------------------------------------------------------------
@dataclass
class SegmentInfo:
    time: Dim
    S: Dim
    L: Dim
    orig_T: int
    pad: int


def segment(
    t: NamedTensor,
    time: Dim,
    into: tuple[Dim, Dim],
    pad_value: float = 0.0,
) -> tuple[NamedTensor, SegmentInfo]:
    """Pad ``time`` to a multiple of ``L`` and split into ``(S, L)``.

    ``S`` binds to ``ceil(T / L)``. The returned ``SegmentInfo`` makes
    ``unsegment`` an exact inverse (crop back to the original T).
    """
    S, L = into
    if L.size is None:
        raise ShapeError("segment: L (chunk length) must be bound")
    T_size = t.size(time)
    s_needed = (T_size + L.size - 1) // L.size
    if S.size is None:
        S.bind(s_needed)
    elif S.size != s_needed:
        from .errors import disagree

        raise disagree(S, S.size, s_needed)
    pad_len = S.size * L.size - T_size
    data = t.data
    if pad_len:
        idx = t.index(time)
        nd = data.dim()
        pad = [0, 0] * nd
        pair_at = 2 * (nd - 1 - idx)
        pad[pair_at + 1] = pad_len
        data = F.pad(data, pad, value=pad_value)
    # Do not rebind the caller's time dim to the padded length.
    merged = ProductDim(S, L)
    tmp_layout = tuple(merged if d is time else d for d in t.layout)
    padded_named = NamedTensor(data, tmp_layout, t.policy, bind=False)
    out = unpack(padded_named, merged, (S, L))
    info = SegmentInfo(time=time, S=S, L=L, orig_T=T_size, pad=pad_len)
    return out, info


def unsegment(t: NamedTensor, info: SegmentInfo) -> NamedTensor:
    """Inverse of ``segment``: merge ``(S, L)`` in place and crop the pad."""
    new_layout: list[Dim] = []
    i = 0
    layout = list(t.layout)
    merged_dim = ProductDim(info.S, info.L)
    while i < len(layout):
        if (
            layout[i] is info.S
            and i + 1 < len(layout)
            and layout[i + 1] is info.L
        ):
            new_layout.append(merged_dim)
            i += 2
            continue
        new_layout.append(layout[i])
        i += 1
    merged = t.to(*new_layout)
    data = merged.data
    if info.pad:
        data = data.narrow(merged.index(merged_dim), 0, info.orig_T)
    out_layout = tuple(info.time if d is merged_dim else d for d in merged.layout)
    return NamedTensor(data, out_layout, t.policy, bind=False)


def expand_to(t: NamedTensor, time: Dim, length: Dim, orig_T: Optional[int] = None) -> NamedTensor:
    """Repeat each position of a leading segment axis ``length`` times, then crop.

    ``t`` has a dim ``S`` (not ``time``); result replaces ``S`` with ``time``.
    """
    # Find the segment axis — first dim that isn't time/length and looks like S
    # API: t layout is (B, S, ...) ; we unsqueeze a L axis, expand, merge, crop.
    if time in t.layout:
        raise ShapeError("expand_to: tensor already has the time dim")
    # assume S is the dim we want to expand — the user passes length=L
    # We need to know which dim is S. Convention: the first non-batch dim,
    # or the dim named S. Cleaner: require the dim to expand is specified.
    raise ShapeError("use expand_legs(t, segment_dim, time, length)")


def expand_legs(
    t: NamedTensor,
    segment: Dim,
    time: Dim,
    length: Dim,
    orig_T: Optional[int] = None,
) -> NamedTensor:
    """``[B, S, ...]`` → ``[B, T, ...]`` by repeating each leg ``length`` times."""
    if length.size is None:
        raise ShapeError("expand_legs: length dim must be bound")
    # unsqueeze L after S, expand, merge S*L, crop
    rest = [d for d in t.layout if d is not segment]
    # insert L after segment
    target_pre = []
    for d in t.layout:
        target_pre.append(d)
        if d is segment:
            target_pre.append(length)
    # to() will insert L as size-1 if unbound... we want size L
    data = t.data
    idx = t.index(segment)
    data = data.unsqueeze(idx + 1)
    sizes = list(data.shape)
    sizes[idx + 1] = length.size
    data = data.expand(*sizes).reshape(
        *data.shape[:idx],
        data.shape[idx] * length.size,
        *data.shape[idx + 2 :],
    )
    T_full = t.size(segment) * length.size
    crop = orig_T if orig_T is not None else T_full
    if crop < T_full:
        data = data.narrow(idx, 0, crop)
    if time.size is None:
        time.bind(crop)
    layout = tuple(time if d is segment else d for d in t.layout)
    return NamedTensor(data, layout, t.policy, bind=False)


# ---------------------------------------------------------------------------
# linear solves
# ---------------------------------------------------------------------------
def solve_triangular(
    system: NamedTensor,
    rhs: NamedTensor,
    *,
    over: tuple[Dim, Dim],
    feature: Dim,
    upper: bool = False,
    unitriangular: bool = False,
) -> NamedTensor:
    """Solve ``system @ x = rhs`` for triangular ``system``.

    ``over = (row, col)`` names the two square axes of ``system``; ``rhs`` is
    indexed by ``row`` plus ``feature``, and the result is indexed by ``col``
    plus ``feature``. Axes present only on one side become broadcast batch axes,
    so a state-batched right-hand side reuses one system without any packing.

    ``unitriangular=True`` assumes a unit diagonal and never reads it. For
    SplitComplex inputs the solve runs in complex64 and the result is cast back
    to the right-hand side dtype.
    """
    from .tensor import _align_to

    row, col = over
    if not system.has(row) or not system.has(col):
        raise ShapeError(
            f"solve_triangular: system {layout_str(system.layout)} needs both "
            f"{row.name} and {col.name}"
        )
    if not rhs.has(row):
        raise missing("solve_triangular", row, "rhs", rhs.layout)
    if not rhs.has(feature):
        raise missing("solve_triangular", feature, "rhs", rhs.layout)

    pair = system.policy.pair if isinstance(system.policy, SplitComplex) else None
    complex_solve = pair is not None and system.has(pair) and rhs.has(pair)

    def _matrix_axes(t: NamedTensor, *keep: Dim) -> list[Dim]:
        skip = {id(d) for d in keep}
        if pair is not None:
            skip.add(id(pair))
        return [d for d in t.layout if id(d) not in skip]

    batch: list[Dim] = _matrix_axes(system, row, col)
    seen = {id(d) for d in batch}
    for d in _matrix_axes(rhs, row, feature):
        if id(d) not in seen:
            batch.append(d)
            seen.add(id(d))

    system_canon = tuple(batch) + (row, col) + ((pair,) if complex_solve else ())
    rhs_canon = tuple(batch) + (row, feature) + ((pair,) if complex_solve else ())
    a = _align_to(system, system_canon)
    b = _align_to(rhs, rhs_canon)

    if complex_solve:
        a = torch.complex(a[..., 0].float(), a[..., 1].float())
        b = torch.complex(b[..., 0].float(), b[..., 1].float())
    solution = torch.linalg.solve_triangular(
        a, b, upper=upper, unitriangular=unitriangular
    )
    out_layout = tuple(batch) + (col, feature)
    if complex_solve:
        target = rhs.dtype
        return as_complex(
            NamedTensor(solution.real.to(target), out_layout, REAL, bind=False),
            NamedTensor(solution.imag.to(target), out_layout, REAL, bind=False),
            pair,
        )
    return NamedTensor(solution, out_layout, rhs.policy, bind=False)


# ---------------------------------------------------------------------------
# gather / Index / at
# ---------------------------------------------------------------------------
class Index:
    """Integer index tensor with a named layout, ranging over a Dim (the table axis)."""

    __slots__ = ("values", "layout", "over")

    def __init__(self, values: torch.Tensor, layout: Sequence[Dim], over: Dim):
        if not values.dtype in (torch.int32, torch.int64, torch.long):
            values = values.long()
        self.values = values
        self.layout = tuple(layout)
        self.over = over
        if values.dim() != len(self.layout):
            raise ShapeError(
                f"Index rank {values.dim()} != layout {layout_str(self.layout)}"
            )

    def __repr__(self) -> str:
        return f"Index{layout_str(self.layout)} over {self.over}"


def gather(t: NamedTensor, over: Dim, index: Union[Index, NamedTensor, torch.Tensor]) -> NamedTensor:
    """Gather ``t`` along ``over`` at integer ``index``."""
    if isinstance(index, Index):
        idx = index.values
        extra = index.layout
        # index layout replaces `over`
    elif isinstance(index, NamedTensor):
        idx = index.data.long()
        extra = index.layout
    else:
        idx = index.long()
        extra = ()
    axis = t.index(over)
    # torch.gather only when the index already has `over` (same rank, same layout).
    index_has_over = isinstance(index, NamedTensor) and index.has(over)
    if index_has_over and idx.dim() == t.data.dim():
        data = torch.gather(t.data, axis, idx)
        return NamedTensor(data, t.layout, t.policy, bind=False)
    # Advanced indexing: table[idx]. Move `over` to 0, then index.
    perm = [axis] + [i for i in range(t.ndim) if i != axis]
    moved = t.data.permute(perm)
    out = moved[idx]
    out_layout = tuple(extra) + tuple(d for d in t.layout if d is not over)
    return NamedTensor(out, out_layout, t.policy, bind=False)


def at(table: NamedTensor, index: Union[Index, NamedTensor, torch.Tensor], over: Optional[Dim] = None) -> NamedTensor:
    """Look up rows of ``table`` at ``index``. ``over`` defaults to table's first dim."""
    axis = over if over is not None else table.layout[0]
    return gather(table, axis, index)


# ---------------------------------------------------------------------------
# scan  h_t = λ ⊙ h_{t-1} + u_t
# ---------------------------------------------------------------------------
def scan(
    u: NamedTensor,
    decay: NamedTensor,
    over: Dim,
    mode: str = "parallel",
    chunk: int = 64,
) -> NamedTensor:
    """Inclusive scan ``h_t = decay * h_{t-1} + u_t`` along ``over``.

    ``decay`` may omit ``over`` (constant λ, LRU-style) or include it
    (per-step γ_t). Modes: ``parallel``, ``chunked``, ``recurrent``.
    """
    if mode not in ("parallel", "chunked", "recurrent"):
        raise ShapeError(f"scan: unknown mode {mode!r}")
    if mode == "recurrent":
        return _scan_recurrent(u, decay, over)
    if mode == "chunked":
        return _scan_chunked(u, decay, over, chunk)
    return _scan_parallel(u, decay, over)


def _decay_for_step(decay: NamedTensor, over: Dim, t_index: int, u: NamedTensor) -> torch.Tensor:
    if decay.has(over):
        # slice that time
        data = decay.to(*[d for d in decay.layout if d is not over], over).data
        # last dim is time
        sl = data[(..., t_index)]
        return sl
    return decay.data


def _scan_recurrent(u: NamedTensor, decay: NamedTensor, over: Dim) -> NamedTensor:
    uu = u.to(*[d for d in u.layout if d is not over], over)
    T = uu.size(over)
    # leading + time last
    data = uu.data
    # h starts 0
    h = torch.zeros_like(data[..., 0])
    outs = []
    dec_const = not decay.has(over)
    if dec_const:
        # align decay onto h's layout (no time)
        h_layout = uu.layout[:-1]
        dec = _align_broadcast(decay, h_layout)
        for t in range(T):
            if isinstance(u.policy, SplitComplex) and u.policy.pair is not None and decay.has(u.policy.pair):
                h = u.policy.mul(dec, h) + data[..., t]
            else:
                h = dec * h + data[..., t]
            outs.append(h)
    else:
        dd = decay.to(*[d for d in decay.layout if d is not over], over)
        for t in range(T):
            sl = NamedTensor(dd.data[..., t], dd.layout[:-1], decay.policy, bind=False)
            dec = _align_broadcast(sl, uu.layout[:-1])
            if isinstance(u.policy, SplitComplex) and u.policy.pair is not None:
                if sl.has(u.policy.pair) if hasattr(sl, "has") else (u.policy.pair in sl.layout):
                    h = u.policy.mul(dec, h) + data[..., t]
                else:
                    h = dec * h + data[..., t]
            else:
                h = dec * h + data[..., t]
            outs.append(h)
    out = torch.stack(outs, dim=-1)
    return NamedTensor(out, uu.layout, u.policy, bind=False)


def _align_broadcast(t: NamedTensor, target_layout: Sequence[Dim]) -> torch.Tensor:
    from .tensor import _align_to

    return _align_to(t, target_layout)


def _align_broadcast_data(data, src: NamedTensor, target_layout):
    return _align_broadcast(NamedTensor(data, src.layout, src.policy, bind=False), target_layout)


def _scan_parallel(u: NamedTensor, decay: NamedTensor, over: Dim) -> NamedTensor:
    """Inclusive scan via the a_t / cumsum trick (real or complex).

    For constant λ (no time on decay): h_t = Σ_{s≤t} λ^{t-s} u_s
    implemented as scale by λ^{-t}, cumsum, scale by λ^{t} when |λ| is safe,
    otherwise fall through to the chunked non-reciprocal form.
    For per-step γ_t: a_t = Π_{k≤t} γ_k, h = a * cumsum(u / a).
    """
    uu = u.to(*[d for d in u.layout if d is not over], over)
    T = uu.size(over)
    if decay.has(over):
        return _scan_per_step_gamma(uu, decay, over)
    # constant λ
    return _scan_chunked(uu, decay, over, chunk=max(T, 1))


def _scan_per_step_gamma(uu: NamedTensor, decay: NamedTensor, over: Dim,
                         chunk: int = 64) -> NamedTensor:
    r"""``h_t = γ_t h_{t-1} + u_t`` with a real per-step ``γ_t``, exact at any length.

    Chunked in log space. With ``L_t = Σ_{k ≤ t} log γ_k`` counted from the
    start of the chunk, inside the chunk

        h_t = Σ_{s ≤ t} e^{L_t - L_s} u_s  +  e^{L_t} h_in ,

    where every exponent is ≤ 0, so nothing overflows or underflows (the old
    ``a · cumsum(u / a)`` form lost all precision once ``Π γ`` fell below
    1e-12, e.g. after ~260 steps of γ = 0.9). The ``[c, c]`` decay matrix is
    built only over the axes the decay actually has; the axes it broadcasts
    over (channels, the complex pair) ride along as matrix columns, so the
    matrix is never copied per channel. Chunks are carried one after another.
    """
    if isinstance(decay.policy, SplitComplex):
        raise ShapeError("scan: a per-step decay must be real")
    x = uu.data                                                  # time last
    g = _align_broadcast(decay.to(*[d for d in decay.layout if d is not over], over),
                         uu.layout)
    nd, T = x.dim(), x.shape[-1]
    lead = list(range(nd - 1))
    carried = [i for i in lead if g.shape[i] == 1 and x.shape[i] != 1]   # decay broadcasts here
    own = [i for i in lead if i not in carried]
    # x -> [own..., T, K], g -> [own..., T]   (K = product of the carried axes)
    xp = x.permute(*own, nd - 1, *carried)
    carried_shape = [x.shape[i] for i in carried]
    K = 1
    for n in carried_shape:
        K *= n
    own_shape = [max(x.shape[i], g.shape[i]) for i in own]
    xp = xp.reshape(*own_shape, T, K)
    gp = g.permute(*own, nd - 1, *carried).reshape(*[g.shape[i] for i in own], T)
    gp = gp.expand(*own_shape, T)

    c = max(1, min(chunk, T))
    pad = (-T) % c
    if pad:
        xp = F.pad(xp, (0, 0, 0, pad))
        gp = F.pad(gp, (0, pad), value=1.0)
    n_chunks = (T + pad) // c
    xs = xp.reshape(*own_shape, n_chunks, c, K)
    logg = torch.log(gp.float().clamp_min(1e-30)).reshape(*own_shape, n_chunks, c)
    L = torch.cumsum(logg, dim=-1)                                # from each chunk start
    diff = L.unsqueeze(-1) - L.unsqueeze(-2)                      # [.., n, t, s]
    lower = torch.ones(c, c, dtype=torch.bool, device=x.device).tril()
    M = torch.where(lower, torch.exp(diff.clamp(max=0.0)), torch.zeros((), device=x.device))
    local = torch.matmul(M.to(xs.dtype), xs)                      # [.., n, c, K]
    from_start = torch.exp(L).to(xs.dtype).unsqueeze(-1)          # [.., n, c, 1]
    h = torch.zeros(*own_shape, K, dtype=xs.dtype, device=x.device)
    blocks = []
    for i in range(n_chunks):
        blk = local[..., i, :, :] + from_start[..., i, :, :] * h.unsqueeze(-2)
        blocks.append(blk)
        h = blk[..., -1, :]
    y = torch.cat(blocks, dim=-2)[..., :T, :]                     # [own..., T, K]
    y = y.reshape(*own_shape, T, *carried_shape)
    inverse = [0] * nd
    for pos, axis in enumerate(own + [nd - 1] + carried):
        inverse[axis] = pos
    y = y.permute(*inverse)
    return NamedTensor(y, uu.layout, uu.policy, bind=False)


def _scan_chunked(u: NamedTensor, decay: NamedTensor, over: Dim, chunk: int) -> NamedTensor:
    """Two-level scan. For constant complex/real λ this matches pands v2 ``diag_scan``.

    When ``decay`` has the pair dim (split-complex λ) and ``u`` is ``[..., T, px]``
    after moving time last, we follow the v2 einsum form (no 1/λ^t).
    """
    uu = u.to(*[d for d in u.layout if d is not over], over)
    if decay.has(over):
        return _scan_per_step_gamma(uu, decay, over)

    pair = uu.policy.pair if isinstance(uu.policy, SplitComplex) else None
    T = uu.size(over)
    c = min(chunk, T)
    if c <= 0:
        c = T

    # Move pair last if present; time is currently last — we want [..., T, px] or [..., T]
    if pair is not None and uu.has(pair):
        uu = uu.to(*[d for d in uu.layout if d is not over and d is not pair], over, pair)
        return _diag_scan_complex(uu, decay, over, pair, c)
    return _diag_scan_real(uu, decay, over, c)


def _diag_scan_real(uu: NamedTensor, decay: NamedTensor, over: Dim, c: int) -> NamedTensor:
    data = uu.data  # [lead..., T]
    T = data.shape[-1]
    lam = _align_broadcast(decay, uu.layout[:-1])  # [lead...]
    pad = (c - T % c) % c
    if pad:
        data = F.pad(data, (0, pad))
    tp = T + pad
    nc = tp // c
    lead = data.shape[:-1]
    u_c = data.reshape(*lead, nc, c)
    idx = torch.arange(c, device=data.device, dtype=data.dtype)
    expo = idx[:, None] - idx[None, :]
    mask = expo >= 0
    expo_m = torch.where(mask, expo, torch.zeros_like(expo))
    log_lam = torch.log(lam.abs().clamp_min(1e-12))
    # log_lam: [lead...]; M[i,j] = lam^(i-j) → [c, c, *lead] via broadcasting on the right
    lead_ones = (1,) * log_lam.ndim
    m = torch.exp(expo_m.reshape(c, c, *lead_ones) * log_lam) * mask.reshape(c, c, *lead_ones)
    intra = torch.einsum("ij...,...nj->...ni", m, u_c)
    # pref[lead..., i] = lam^(i+1)
    pref = torch.exp((idx + 1.0) * log_lam.unsqueeze(-1))
    intra_last = intra[..., -1]
    cidx = torch.arange(nc, device=data.device, dtype=data.dtype)
    cexpo = cidx[:, None] - 1.0 - cidx[None, :]
    cmask = cexpo >= 0
    cexpo_m = torch.where(cmask, cexpo, torch.zeros_like(cexpo))
    g = torch.exp((cexpo_m * c).reshape(nc, nc, *lead_ones) * log_lam) * cmask.reshape(
        nc, nc, *lead_ones
    )
    state_start = torch.einsum("xy...,...y->...x", g, intra_last)
    h_c = intra + pref.unsqueeze(-2) * state_start.unsqueeze(-1)
    h = h_c.reshape(*lead, tp)[..., :T]
    return NamedTensor(h, uu.layout, uu.policy, bind=False)


def _diag_scan_complex(
    uu: NamedTensor, decay: NamedTensor, over: Dim, pair: Dim, c: int
) -> NamedTensor:
    """pands v2 ``diag_scan``: u [..., T, 2], lam [..., 2] (usually [D, 2])."""
    u = uu.data
    # decay should have pair, no time
    dec = decay.to(*[d for d in decay.layout if d is not pair], pair) if decay.has(pair) else decay
    # Align feature dims: uu is [batch..., T, px], decay is [feat..., px]
    # Typical: u [B,T,D,2] after we put T before px — wait we did to(..., T, px)
    # so layout is [rest..., T, px] and rest includes D.
    # decay is [D, px] or [px] or [rest..., px]
    orig_dtype = u.dtype
    b_shape = u.shape[:-2]  # everything except T, px
    t = u.shape[-2]
    # Identify feature dim(s) between batch and T — we treat all but last-2 as lead
    # For v2 compatibility the lead is [B, D] if u was [B, T, D, 2] then to(B, D, T, px)
    # Caller should put feature dims before time.
    with torch.autocast(device_type="cuda", enabled=False):
        u_f = u.float()
        # broadcast lam onto [..., 2] matching u's leading dims except T
        lam_t = _align_broadcast(dec, [d for d in uu.layout if d is not over])
        # lam_t shape: [lead..., 2]
        lam_f = lam_t.float()
        lam_c = torch.complex(lam_f[..., 0], lam_f[..., 1])
        log_lam = torch.log(lam_c.abs().clamp_min(1e-12)) + 1j * torch.angle(lam_c)

        pad = (c - t % c) % c
        if pad:
            # pad time
            u_f = F.pad(u_f, (0, 0, 0, pad))
        tp = t + pad
        nc = tp // c
        u_c = torch.complex(u_f[..., 0], u_f[..., 1]).reshape(*u_f.shape[:-2], nc, c)
        # u_c: [lead..., nc, c]
        idx = torch.arange(c, device=u.device, dtype=torch.float32)
        expo = idx[:, None] - idx[None, :]
        mask = expo >= 0
        expo_m = torch.where(mask, expo, torch.zeros_like(expo))
        # log_lam: [lead...]
        m_mat = torch.exp(expo_m.reshape(c, c, *([1] * log_lam.ndim)) * log_lam) * mask.reshape(
            c, c, *([1] * log_lam.ndim)
        )
        intra = torch.einsum("ij...,...nj->...ni", m_mat, u_c)
        pref = torch.exp((idx + 1.0) * log_lam.unsqueeze(-1))
        intra_last = intra[..., -1]
        cidx = torch.arange(nc, device=u.device, dtype=torch.float32)
        cexpo = cidx[:, None] - 1.0 - cidx[None, :]
        cmask = cexpo >= 0
        cexpo_m = torch.where(cmask, cexpo, torch.zeros_like(cexpo))
        g_mat = torch.exp((cexpo_m * c).reshape(nc, nc, *([1] * log_lam.ndim)) * log_lam) * cmask.reshape(
            nc, nc, *([1] * log_lam.ndim)
        )
        state_start = torch.einsum("xy...,...y->...x", g_mat, intra_last)
        h_c = intra + pref.unsqueeze(-2) * state_start.unsqueeze(-1)
        h_c = h_c.reshape(*u_f.shape[:-2], tp)[..., :t]
        h = torch.stack([h_c.real, h_c.imag], dim=-1)
    return NamedTensor(h.to(dtype=orig_dtype), uu.layout, uu.policy, bind=False)
