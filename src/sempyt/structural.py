"""Named structural + elementwise ops — never a numeric ``dim=`` in model code.

Every op that torch spells with an integer axis (``stack``/``cat``/``split``/
``cumsum``/``topk``/``narrow``/``repeat_interleave`` ...) is re-expressed here
over Dim objects. Think in N axes: name the axis you act on, not "row/column".
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence, Union

import torch

from .dim import Dim
from .errors import ShapeError, disagree, missing
from .ops import Index
from .tensor import NamedTensor, named

DimOrName = Union[Dim, str]


def _resolve(t: NamedTensor, d: DimOrName) -> Dim:
    if isinstance(d, Dim):
        return d
    for dim in t.layout:
        if dim.name == d:
            return dim
    raise missing("op", Dim(d), "tensor", t.layout)


# --------------------------------------------------------------------------- #
# stack / cat / split — build or cut a named axis
# --------------------------------------------------------------------------- #
def stack(tensors: Sequence[NamedTensor], into: Dim, at: Optional[Dim] = None) -> NamedTensor:
    """Stack same-layout tensors along a NEW named axis ``into`` (size = len).

    ``at`` (optional) names an existing dim to insert *before*; default appends.
    """
    if not tensors:
        raise ShapeError("stack: empty list")
    ref = tensors[0]
    for t in tensors[1:]:
        if [id(d) for d in t.layout] != [id(d) for d in ref.layout]:
            raise ShapeError("stack: all tensors must share the exact layout")
    if into.size not in (None, len(tensors)):
        raise disagree(into, into.size, len(tensors))
    into.bind(len(tensors))
    pos = len(ref.layout) if at is None else ref.index(_resolve(ref, at))
    data = torch.stack([t.data for t in tensors], dim=pos)
    layout = list(ref.layout)
    layout.insert(pos, into)
    return NamedTensor(data, layout, ref.policy, bind=False)


def unstack(t: NamedTensor, over: DimOrName) -> list[NamedTensor]:
    """Inverse of ``stack``: a Python list of slices, ``over`` removed."""
    d = _resolve(t, over)
    axis = t.index(d)
    layout = tuple(x for x in t.layout if x is not d)
    return [NamedTensor(chunk.squeeze(axis), layout, t.policy, bind=False)
            for chunk in t.data.split(1, dim=axis)]


def cat(
    tensors: Sequence[NamedTensor],
    over: DimOrName,
    into: Optional[Dim] = None,
) -> NamedTensor:
    """Concatenate along an existing named axis ``over``.

    The concatenated axis is matched **by name**, so pieces may carry their own
    Dim object of that name with their own size (chunked loops whose last chunk
    is short). All other axes must be the same Dim identities as the first piece.
    ``into`` names the joined axis (e.g. chunks back to the full ``time``);
    by default a fresh Dim of the same name is created.
    """
    if not tensors:
        raise ShapeError("cat: empty list")
    ref = tensors[0]
    # Resolve by name so a caller holding one chunk's Dim can join all of them.
    d = _resolve(ref, over.name if isinstance(over, Dim) else over)
    axis = ref.index(d)
    aligned = []
    for t in tensors:
        own = _resolve(t, d.name)
        target = tuple(own if x is d else x for x in ref.layout)
        aligned.append(t.to(*target).data)
    data = torch.cat(aligned, dim=axis)
    if into is not None:
        if into.size is not None and into.size != data.shape[axis]:
            raise disagree(into, into.size, data.shape[axis])
        into.bind(data.shape[axis])
        new_d = into
    else:
        # concatenation produces a fresh dim of summed size; keep the name
        new_d = Dim(d.name, data.shape[axis], role=d.role)
    layout = tuple(new_d if x is d else x for x in ref.layout)
    return NamedTensor(data, layout, ref.policy, bind=False)


def split(t: NamedTensor, over: DimOrName, sizes: Sequence[int], names: Sequence[Dim]) -> list[NamedTensor]:
    """Split a named axis into pieces of ``sizes`` (each a new named dim)."""
    d = _resolve(t, over)
    axis = t.index(d)
    if sum(sizes) != d.size:
        raise ShapeError(f"split: sizes {list(sizes)} sum to {sum(sizes)} != {d.name}({d.size})")
    if len(names) != len(sizes):
        raise ShapeError("split: need one name per piece")
    parts = t.data.split(list(sizes), dim=axis)
    out = []
    for part, nm in zip(parts, names):
        nm.bind(part.shape[axis])
        layout = tuple(nm if x is d else x for x in t.layout)
        out.append(NamedTensor(part, layout, t.policy, bind=False))
    return out


def flatten(t: NamedTensor, dims: Sequence[DimOrName], into: Dim) -> NamedTensor:
    """Merge several named axes into one, placed where the first of them was.

    The merged axes are moved to be adjacent (in the order given) if they are
    not already, then reshaped into ``into``. ``into`` binds to the product of
    their sizes. Inverse of :func:`unflatten`.
    """
    resolved = [_resolve(t, d) for d in dims]
    dim_ids = {id(d) for d in resolved}
    positions = [t.index(d) for d in resolved]
    first_pos = min(positions)
    last_pos = max(positions)
    before = [d for d in t.layout[:first_pos] if id(d) not in dim_ids]
    after = [d for d in t.layout[last_pos + 1:] if id(d) not in dim_ids]
    target = tuple(before) + tuple(resolved) + tuple(after)
    moved = t.to(*target)
    pos = len(before)
    n = len(resolved)
    total = 1
    for d in resolved:
        if d.size is None:
            raise ShapeError(f"flatten: {d.name} has no bound size")
        total *= d.size
    if into.size is None:
        into.bind(total)
    elif into.size != total:
        raise disagree(into, into.size, total)
    data = moved.data.reshape(*moved.data.shape[:pos], total, *moved.data.shape[pos + n:])
    layout = tuple(before) + (into,) + tuple(after)
    return NamedTensor(data, layout, t.policy, bind=False)


def unflatten(t: NamedTensor, over: DimOrName, into: Sequence[Dim]) -> NamedTensor:
    """Split one named axis into several (inverse of :func:`flatten`).

    Each dim in ``into`` must have a bound size; their product must equal
    ``over``'s size. The new axes replace ``over`` in place.
    """
    d = _resolve(t, over)
    axis = t.index(d)
    if d.size is None:
        raise ShapeError(f"unflatten: {d.name} has no bound size")
    sizes = []
    for x in into:
        if x.size is None:
            raise ShapeError(f"unflatten: {x.name} has no bound size")
        sizes.append(x.size)
    if math.prod(sizes) != d.size:
        raise ShapeError(
            f"unflatten: {sizes} product {math.prod(sizes)} != {d.name}({d.size})"
        )
    data = t.data.reshape(*t.data.shape[:axis], *sizes, *t.data.shape[axis + 1:])
    layout = list(t.layout)
    layout[axis:axis + 1] = list(into)
    return NamedTensor(data, tuple(layout), t.policy, bind=False)


def select(t: NamedTensor, over: DimOrName, index: int) -> NamedTensor:
    """Pick one integer position along ``over``; that axis is dropped."""
    d = _resolve(t, over)
    axis = t.index(d)
    data = t.data.select(axis, index)
    layout = tuple(x for x in t.layout if x is not d)
    return NamedTensor(data, layout, t.policy, bind=False)


def take(t: NamedTensor, over: DimOrName, start: int, length: int, new: Optional[Dim] = None) -> NamedTensor:
    """Slice a range ``[start, start+length)`` along ``over`` (torch ``narrow``)."""
    d = _resolve(t, over)
    axis = t.index(d)
    data = t.data.narrow(axis, start, length)
    nd = new if new is not None else Dim(d.name, length, role=d.role)
    nd.bind(length)
    layout = tuple(nd if x is d else x for x in t.layout)
    return NamedTensor(data, layout, t.policy, bind=False)


# --------------------------------------------------------------------------- #
# cumulative / argmax / topk
# --------------------------------------------------------------------------- #
def cumsum(t: NamedTensor, over: DimOrName) -> NamedTensor:
    d = _resolve(t, over)
    return NamedTensor(torch.cumsum(t.data, dim=t.index(d)), t.layout, t.policy, bind=False)


def cumprod(t: NamedTensor, over: DimOrName) -> NamedTensor:
    d = _resolve(t, over)
    return NamedTensor(torch.cumprod(t.data, dim=t.index(d)), t.layout, t.policy, bind=False)


def argmax(t: NamedTensor, over: DimOrName) -> Index:
    """Argmax along ``over`` → an ``Index`` (integers) over that axis."""
    d = _resolve(t, over)
    idx = torch.argmax(t.data, dim=t.index(d))
    layout = tuple(x for x in t.layout if x is not d)
    return Index(idx, layout, over=d)


def argmin(t: NamedTensor, over: DimOrName) -> Index:
    d = _resolve(t, over)
    idx = torch.argmin(t.data, dim=t.index(d))
    layout = tuple(x for x in t.layout if x is not d)
    return Index(idx, layout, over=d)


def topk(t: NamedTensor, over: DimOrName, k: int, kdim: Dim) -> tuple[NamedTensor, Index]:
    """Top-``k`` along ``over``. Returns (values with axis ``over``→``kdim``, Index).

    Both outputs share the same layout (``over`` replaced by ``kdim``); the Index
    still records that its integers point into the original ``over`` axis.
    """
    d = _resolve(t, over)
    axis = t.index(d)
    vals, idx = torch.topk(t.data, k, dim=axis)
    kdim.bind(k)
    vlayout = tuple(kdim if x is d else x for x in t.layout)
    return (
        NamedTensor(vals, vlayout, t.policy, bind=False),
        Index(idx, vlayout, over=d),
    )


# --------------------------------------------------------------------------- #
# repeat / expand / flip / roll / pad
# --------------------------------------------------------------------------- #
def repeat_along(t: NamedTensor, over: DimOrName, times: int, new: Optional[Dim] = None) -> NamedTensor:
    """``repeat_interleave`` along ``over`` (GQA head expansion)."""
    d = _resolve(t, over)
    axis = t.index(d)
    data = t.data.repeat_interleave(times, dim=axis)
    nd = new if new is not None else Dim(d.name, d.size * times, role=d.role)
    nd.bind(d.size * times)
    layout = tuple(nd if x is d else x for x in t.layout)
    return NamedTensor(data, layout, t.policy, bind=False)


def flip(t: NamedTensor, over: DimOrName) -> NamedTensor:
    d = _resolve(t, over)
    return NamedTensor(torch.flip(t.data, dims=[t.index(d)]), t.layout, t.policy, bind=False)


def shift(t: NamedTensor, over: DimOrName, by: int = 1, fill: float = 0.0) -> NamedTensor:
    """Non-circular shift along ``over`` by ``by`` steps, zero-filling the gap.

    ``by > 0`` moves values toward larger indices (a causal "look back one").
    """
    d = _resolve(t, over)
    axis = t.index(d)
    data = torch.roll(t.data, shifts=by, dims=axis).clone()
    idx = [slice(None)] * t.data.dim()
    idx[axis] = slice(0, by) if by > 0 else slice(by, None)
    data[tuple(idx)] = fill
    return NamedTensor(data, t.layout, t.policy, bind=False)


def roll(t: NamedTensor, over: DimOrName, shift: int) -> NamedTensor:
    d = _resolve(t, over)
    return NamedTensor(torch.roll(t.data, shifts=shift, dims=t.index(d)), t.layout, t.policy, bind=False)


def pad(t: NamedTensor, over: DimOrName, left: int = 0, right: int = 0, value: float = 0.0,
        new: Optional[Dim] = None) -> NamedTensor:
    """Zero-(or value-)pad ``over`` on left/right; produces a fresh sized dim."""
    d = _resolve(t, over)
    axis = t.index(d)
    nd = t.data.dim()
    spec = [0, 0] * nd
    at = 2 * (nd - 1 - axis)
    spec[at] = left
    spec[at + 1] = right
    data = torch.nn.functional.pad(t.data, spec, value=value)
    ndim = new if new is not None else Dim(d.name, d.size + left + right, role=d.role)
    ndim.bind(d.size + left + right)
    layout = tuple(ndim if x is d else x for x in t.layout)
    return NamedTensor(data, layout, t.policy, bind=False)


# --------------------------------------------------------------------------- #
# where + unary elementwise map (layout-preserving)
# --------------------------------------------------------------------------- #
def where(cond: NamedTensor, a: NamedTensor, b: NamedTensor) -> NamedTensor:
    """Named select. All three broadcast on the union of their layouts."""
    from .tensor import _align_to

    canon = list(a.layout)
    seen = {id(d) for d in canon}
    for src in (b, cond):
        for d in src.layout:
            if id(d) not in seen:
                canon.append(d)
                seen.add(id(d))
    ca = _align_to(cond, canon)
    aa = _align_to(a, canon)
    ba = _align_to(b, canon)
    return NamedTensor(torch.where(ca.bool(), aa, ba), tuple(canon), a.policy, bind=False)


def _compare(a: NamedTensor, b, op) -> NamedTensor:
    from .tensor import _align_to

    if isinstance(b, NamedTensor):
        canon = list(a.layout)
        seen = {id(d) for d in canon}
        for d in b.layout:
            if id(d) not in seen:
                canon.append(d)
                seen.add(id(d))
        aa = _align_to(a, canon)
        bb = _align_to(b, canon)
        return NamedTensor(op(aa, bb), tuple(canon), a.policy, bind=False)
    return NamedTensor(op(a.data, b), a.layout, a.policy, bind=False)


def eq(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x == y)


def ne(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x != y)


def ge(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x >= y)


def gt(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x > y)


def le(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x <= y)


def lt(a: NamedTensor, b) -> NamedTensor:
    return _compare(a, b, lambda x, y: x < y)


def l2_normalize(t: NamedTensor, over: DimOrName, eps: float = 1e-12) -> NamedTensor:
    """Unit-norm along a named axis (keeps layout)."""
    d = _resolve(t, over)
    axis = t.index(d)
    norm = t.data.pow(2).sum(dim=axis, keepdim=True).clamp_min(eps).sqrt()
    return NamedTensor(t.data / norm, t.layout, t.policy, bind=False)


def apply(t: NamedTensor, fn: Callable[[torch.Tensor], torch.Tensor]) -> NamedTensor:
    """Apply a layout-preserving elementwise torch fn."""
    return NamedTensor(fn(t.data), t.layout, t.policy, bind=False)


_UNARY = {
    "cos": torch.cos, "sin": torch.sin, "exp": torch.exp, "log": torch.log,
    "sqrt": torch.sqrt, "tanh": torch.tanh, "sigmoid": torch.sigmoid,
    "relu": torch.relu, "silu": torch.nn.functional.silu,
    "gelu": torch.nn.functional.gelu, "softplus": torch.nn.functional.softplus,
    "abs_real": torch.abs, "square": torch.square,
}


def _make_unary(name, torch_fn):
    def op(t: NamedTensor) -> NamedTensor:
        return NamedTensor(torch_fn(t.data), t.layout, t.policy, bind=False)

    op.__name__ = name
    op.__doc__ = f"Elementwise {name}, layout preserved."
    return op


cos = _make_unary("cos", torch.cos)
sin = _make_unary("sin", torch.sin)
exp = _make_unary("exp", torch.exp)
log = _make_unary("log", torch.log)
sqrt = _make_unary("sqrt", torch.sqrt)
tanh = _make_unary("tanh", torch.tanh)
sigmoid = _make_unary("sigmoid", torch.sigmoid)
relu = _make_unary("relu", torch.relu)
silu = _make_unary("silu", torch.nn.functional.silu)
gelu = _make_unary("gelu", torch.nn.functional.gelu)
softplus = _make_unary("softplus", torch.nn.functional.softplus)
square = _make_unary("square", torch.square)


def rotate(x: NamedTensor, angle: NamedTensor, pair: DimOrName) -> NamedTensor:
    r"""Turn every 2-vector along ``pair`` by ``angle`` radians.

        (a, b)  ->  (a cos θ - b sin θ,  a sin θ + b cos θ)

    This is ``z · e^{iθ}`` written over real numbers: RoPE, a learned
    rotary clock and a phase-weighted read are all this one op. ``x`` holds
    one plane per position on a size-2 ``pair`` axis; ``angle`` carries any
    subset of ``x``'s other axes and broadcasts by name. When ``angle`` adds
    no new axis, the result keeps ``x``'s layout.
    """
    p = _resolve(x, pair)
    if p.size != 2:
        raise ShapeError(f"rotate: pair axis {p} must have size 2")
    a, b = select(x, p, 0), select(x, p, 1)
    c, s = cos(angle), sin(angle)
    turned = stack([a * c - b * s, a * s + b * c], into=p)
    if {id(d) for d in turned.layout} == {id(d) for d in x.layout}:
        return turned.to(*x.layout)
    return turned


def clamp(t: NamedTensor, min=None, max=None) -> NamedTensor:  # noqa: A002
    return NamedTensor(t.data.clamp(min=min, max=max), t.layout, t.policy, bind=False)


def zeros_like(t: NamedTensor) -> NamedTensor:
    return NamedTensor(torch.zeros_like(t.data), t.layout, t.policy, bind=False)


# --------------------------------------------------------------------------- #
# constructors — build a tensor from named axes, never from a shape tuple
# --------------------------------------------------------------------------- #
def _sizes(layout: Sequence[Dim]) -> list[int]:
    out = []
    for d in layout:
        if d.size is None:
            raise ShapeError(f"constructor: {d.name} has no bound size")
        out.append(d.size)
    return out


def full(value: float, *layout: Dim, policy=None, device=None, dtype=None) -> NamedTensor:
    """Constant tensor shaped by named axes: ``full(0.0, batch, heads, time)``."""
    data = torch.full(_sizes(layout), value, device=device, dtype=dtype)
    return NamedTensor(data, layout, policy, bind=False)


def zeros(*layout: Dim, policy=None, device=None, dtype=None) -> NamedTensor:
    """Zeros shaped by named axes (replaces ``torch.zeros`` + shape bookkeeping)."""
    data = torch.zeros(_sizes(layout), device=device, dtype=dtype)
    return NamedTensor(data, layout, policy, bind=False)


def ones(*layout: Dim, policy=None, device=None, dtype=None) -> NamedTensor:
    """Ones shaped by named axes."""
    data = torch.ones(_sizes(layout), device=device, dtype=dtype)
    return NamedTensor(data, layout, policy, bind=False)


def arange(over: Dim, *, device=None, dtype=None) -> NamedTensor:
    """Positions ``0..over.size-1`` as a tensor on the ``over`` axis."""
    if over.size is None:
        raise ShapeError(f"arange: {over.name} has no bound size")
    data = torch.arange(over.size, device=device, dtype=dtype)
    return NamedTensor(data, (over,), None, bind=False)


def eye(rows: Dim, cols: Dim, *, device=None, dtype=None) -> NamedTensor:
    """Identity matrix on two named axes of equal size."""
    if rows.size != cols.size:
        raise disagree(rows, rows.size, cols.size, other=cols)
    data = torch.eye(rows.size, device=device, dtype=dtype)
    return NamedTensor(data, (rows, cols), None, bind=False)


def tril_mask(rows: Dim, cols: Dim, *, diagonal: int = 0, device=None, dtype=None) -> NamedTensor:
    """Lower-triangular 0/1 mask on ``(rows, cols)``.

    ``diagonal=0`` keeps the diagonal (causal-inclusive); ``diagonal=-1`` drops
    it (strictly-past). Named replacement for ``torch.tril(torch.ones(...))``.
    """
    data = torch.tril(
        torch.ones(rows.size, cols.size, device=device, dtype=dtype), diagonal
    )
    return NamedTensor(data, (rows, cols), None, bind=False)


def full_like(t: NamedTensor, value: float) -> NamedTensor:
    return NamedTensor(torch.full_like(t.data, value), t.layout, t.policy, bind=False)
