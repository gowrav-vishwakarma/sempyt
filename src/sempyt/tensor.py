"""NamedTensor: a torch.Tensor + a named Layout + an element Policy.

Frontend-only: pytree-registered dataclass, not a Tensor subclass. ``.raw()``
is the explicit exit to kernels / SDPA / Triton.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
from torch.utils import _pytree as pytree

from .dim import Dim, ProductDim, parse_layout, lookup_from
from .errors import ShapeError, missing
from .layout import (
    ShapeMap,
    bind_layout,
    layout_str,
    plan_to,
    resolve_spec,
)
from .policies import REAL, ElementPolicy, Real, SplitComplex


class NamedTensor:
    """Wraps a real ``torch.Tensor`` + ordered Dims + a policy."""

    __slots__ = ("data", "layout", "policy")

    def __init__(
        self,
        data: torch.Tensor,
        layout: Sequence[Dim],
        policy: Optional[ElementPolicy] = None,
        *,
        bind: bool = True,
    ):
        if not torch.is_tensor(data):
            data = torch.as_tensor(data)
        self.data = data
        self.layout = tuple(layout)
        self.policy = policy if policy is not None else REAL
        if len(self.layout) != data.dim():
            raise ShapeError(
                f"rank mismatch: tensor has {data.dim()} dims {tuple(data.shape)}, "
                f"layout {layout_str(self.layout)} has {len(self.layout)}"
            )
        if bind:
            bind_layout(self.layout, data.shape)

    # -- shape ----------------------------------------------------------------
    @property
    def shape(self) -> ShapeMap:
        return ShapeMap(self.layout, self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return self.data.device

    @property
    def ndim(self) -> int:
        return self.data.dim()

    def size(self, dim: Dim) -> int:
        return self.shape[dim]

    def has(self, dim: Dim) -> bool:
        return any(d is dim for d in self.layout)

    def index(self, dim: Dim) -> int:
        for i, d in enumerate(self.layout):
            if d is dim:
                return i
        raise missing("index", dim, "tensor", self.layout)

    # -- layout algebra -------------------------------------------------------
    def to(self, *spec) -> "NamedTensor":
        """Re-express this tensor under a new named layout.

        Accepts Dim objects (``x.to(B, H, T, d)``) or a single string
        (``x.to("B H T d")`` / ``x.to("B (H d)")``).
        A no-op (same objects, same order) returns ``self``.
        """
        if len(spec) == 1 and isinstance(spec[0], str):
            new_layout = parse_layout(spec[0], lookup_from(self.layout))
        elif len(spec) == 1 and not isinstance(spec[0], Dim) and _is_layout_seq(spec[0]):
            new_layout = tuple(spec[0])
        else:
            new_layout = tuple(spec)
        plan = plan_to(self.layout, new_layout, self.data.shape)
        if plan.identity:
            return self
        return NamedTensor(plan.apply(self.data), new_layout, self.policy, bind=False)

    def raw(self, *spec, contiguous: bool = True) -> torch.Tensor:
        """Exit to a plain torch tensor in the requested axis order.

        Mandatory before FlashAttention / SDPA / Triton / ``solve_triangular``.
        """
        t = self.to(*spec) if spec else self
        data = t.data
        if contiguous and not data.is_contiguous():
            data = data.contiguous()
        return data

    def alias(self, old: Dim, new: Dim) -> "NamedTensor":
        """Replace ``old`` with ``new`` in the layout without moving data.

        Sizes are correlated. Used when the same physical axis must appear
        twice under two identities (e.g. attention ``Tq`` vs ``Tk``).
        """
        if new.size is None and old.size is not None:
            new.bind(old.size)
        elif old.size is not None and new.size is not None and old.size != new.size:
            from .errors import disagree

            raise disagree(old, old.size, new.size, other=new)
        layout = tuple(new if d is old else d for d in self.layout)
        return NamedTensor(self.data, layout, self.policy, bind=False)

    def relabel(self, **mapping) -> "NamedTensor":
        """``x.relabel()`` is alias by name; prefer ``alias(old, new)``."""
        raise ShapeError("use alias(old_dim, new_dim) — relabel-by-name is not identity-safe")

    # -- policy-level element ops ---------------------------------------------
    def conj(self) -> "NamedTensor":
        return NamedTensor(self.policy.conj(self.data), self.layout, self.policy, bind=False)

    def abs(self) -> "NamedTensor":
        data = self.policy.abs(self.data)
        if isinstance(self.policy, SplitComplex) and self.policy.pair is not None:
            layout = tuple(d for d in self.layout if d is not self.policy.pair)
            return NamedTensor(data, layout, REAL, bind=False)
        return NamedTensor(data, self.layout, self.policy, bind=False)

    def normalize(self, over: Dim, eps: Optional[float] = None) -> "NamedTensor":
        idx = self.index(over)
        kwargs = {} if eps is None else {"eps": eps}
        out = self.policy.norm_vec(self.data, idx, **kwargs)
        return NamedTensor(out, self.layout, self.policy, bind=False)

    # -- operators (named broadcast) ------------------------------------------
    def __add__(self, other):
        return _binop(self, other, "add")

    def __radd__(self, other):
        return _binop(self, other, "add")

    def __sub__(self, other):
        return _binop(self, other, "sub")

    def __rsub__(self, other):
        if _is_scalar(other):
            return NamedTensor(other - self.data, self.layout, self.policy, bind=False)
        return _binop(_as_named(other, self), self, "sub")

    def __mul__(self, other):
        return _binop(self, other, "mul")

    def __rmul__(self, other):
        return _binop(self, other, "mul")

    def __truediv__(self, other):
        if _is_scalar(other):
            return NamedTensor(self.data / other, self.layout, self.policy, bind=False)
        other_n = _as_named(other, self)
        xa, ya, canon, pol = _broadcast_pair(self, other_n)
        return NamedTensor(xa / ya, canon, pol, bind=False)

    def __neg__(self):
        return NamedTensor(self.policy.neg(self.data), self.layout, self.policy, bind=False)

    def __matmul__(self, other):
        from .ops import contract

        if not isinstance(other, NamedTensor):
            raise TypeError("@ on NamedTensor requires another NamedTensor")
        shared = [d for d in self.layout if any(d is e for e in other.layout)]
        if not shared:
            raise ShapeError("@ needs a shared Dim to contract over")
        return contract(self, other, over=shared[-1])

    def with_policy(self, policy) -> "NamedTensor":
        """Reinterpret the same data under a different element policy."""
        return NamedTensor(self.data, self.layout, policy, bind=False)

    # -- named structural sugar (delegates to sempyt.structural) --------------
    def select(self, over, index: int) -> "NamedTensor":
        from .structural import select

        return select(self, over, index)

    def __getitem__(self, key) -> "NamedTensor":
        """``x[dim, i]`` selects position ``i`` along ``dim`` (drops the axis)."""
        if isinstance(key, tuple) and len(key) == 2:
            return self.select(key[0], key[1])
        raise TypeError("index a NamedTensor as x[dim, i]; other forms use named ops")

    def take(self, over, start: int, length: int, new=None) -> "NamedTensor":
        from .structural import take

        return take(self, over, start, length, new)

    def cumsum(self, over) -> "NamedTensor":
        from .structural import cumsum

        return cumsum(self, over)

    def cumprod(self, over) -> "NamedTensor":
        from .structural import cumprod

        return cumprod(self, over)

    def flip(self, over) -> "NamedTensor":
        from .structural import flip

        return flip(self, over)

    def roll(self, over, shift: int) -> "NamedTensor":
        from .structural import roll

        return roll(self, over, shift)

    def shift(self, over, by: int = 1, fill: float = 0.0) -> "NamedTensor":
        from .structural import shift

        return shift(self, over, by, fill)

    def repeat_along(self, over, times: int, new=None) -> "NamedTensor":
        from .structural import repeat_along

        return repeat_along(self, over, times, new)

    # -- torch interop --------------------------------------------------------
    def float(self) -> "NamedTensor":
        return NamedTensor(self.data.float(), self.layout, self.policy, bind=False)

    def to_dtype(self, dtype) -> "NamedTensor":
        return NamedTensor(self.data.to(dtype), self.layout, self.policy, bind=False)

    def clone(self) -> "NamedTensor":
        return NamedTensor(self.data.clone(), self.layout, self.policy, bind=False)

    def detach(self) -> "NamedTensor":
        return NamedTensor(self.data.detach(), self.layout, self.policy, bind=False)

    def __repr__(self) -> str:
        return f"NamedTensor{layout_str(self.layout)} {tuple(self.data.shape)} {self.policy}"


def named(
    data,
    layout: Sequence[Dim],
    policy: Optional[ElementPolicy] = None,
) -> NamedTensor:
    """Build a NamedTensor. ``data`` is a real torch tensor (or array)."""
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    return NamedTensor(data, layout, policy)


def _is_layout_seq(obj) -> bool:
    if isinstance(obj, (str, Dim, NamedTensor)):
        return False
    try:
        return all(isinstance(x, Dim) for x in obj)
    except TypeError:
        return False


def _is_scalar(x) -> bool:
    if isinstance(x, (int, float)):
        return True
    if torch.is_tensor(x) and x.ndim == 0:
        return True
    return False


def _as_named(other, like: NamedTensor) -> NamedTensor:
    if isinstance(other, NamedTensor):
        return other
    if torch.is_tensor(other) and other.ndim == 0:
        # broadcast scalar later
        raise TypeError("use a Python scalar or a NamedTensor")
    raise TypeError(f"expected NamedTensor, got {type(other)}")


def _broadcast_pair(x: NamedTensor, y: NamedTensor):
    """Align two tensors onto the union of their layouts (x order + y-only)."""
    # Correlate shared sizes
    y_by_id = {id(d): d for d in y.layout}
    for dx in x.layout:
        dy = y_by_id.get(id(dx))
        if dy is None:
            continue
        if dx.size is not None and dy.size is not None and dx.size != dy.size:
            from .errors import disagree

            raise disagree(dx, dx.size, dy.size, other=dy)
    canon = list(x.layout)
    seen = {id(d) for d in canon}
    for dy in y.layout:
        if id(dy) not in seen:
            canon.append(dy)
            seen.add(id(dy))

    # Missing axes become size-1 (torch broadcasts). Never use to() here —
    # to() refuses to insert a dim whose bound size is not 1.
    xa = _align_to(x, canon)
    ya = _align_to(y, canon)

    pol = x.policy
    if isinstance(y.policy, SplitComplex) and not isinstance(x.policy, SplitComplex):
        pol = y.policy
    return xa, ya, tuple(canon), pol


def _align_to(t: NamedTensor, canon: Sequence[Dim]) -> torch.Tensor:
    """Permute / unsqueeze ``t`` onto ``canon``, broadcasting missing axes as 1."""
    # If t already matches, return data
    if len(t.layout) == len(canon) and all(a is b for a, b in zip(t.layout, canon)):
        return t.data
    # Build a target layout that uses size-1 stand-ins... we unsqueeze missing
    # then permute.
    present = []
    data = t.data
    layout = list(t.layout)
    for d in canon:
        if any(e is d for e in layout):
            continue
        # missing: unsqueeze at end, record
        data = data.unsqueeze(-1)
        layout.append(d)
    # permute layout → canon
    pos = {id(d): i for i, d in enumerate(layout)}
    perm = [pos[id(d)] for d in canon]
    if perm != list(range(len(perm))):
        data = data.permute(perm)
    # expand to broadcast sizes (torch will do this in the op; but make
    # shapes compatible by expand if needed — the op can broadcast)
    return data


def _binop(x: NamedTensor, other, op: str) -> NamedTensor:
    if _is_scalar(other):
        if op == "add":
            return NamedTensor(x.data + other, x.layout, x.policy, bind=False)
        if op == "sub":
            return NamedTensor(x.data - other, x.layout, x.policy, bind=False)
        if op == "mul":
            return NamedTensor(x.data * other, x.layout, x.policy, bind=False)
    if not isinstance(other, NamedTensor):
        raise TypeError(f"{op} expected NamedTensor or scalar, got {type(other)}")
    xa, ya, canon, pol = _broadcast_pair(x, other)
    # Complex * complex uses policy.mul (cmul). Real * anything is torch *.
    # Real * split-complex (broadcast, no pair on the real side): torch *.
    both_complex = isinstance(x.policy, SplitComplex) and isinstance(other.policy, SplitComplex)
    if op == "mul" and both_complex:
        pair = x.policy.pair
        if pair is not None and any(d is pair for d in canon):
            last = tuple(d for d in canon if d is not pair) + (pair,)
            xa = _align_to(x, last)
            ya = _align_to(other, last)
            out = pol.mul(xa, ya)
            return NamedTensor(out, last, pol, bind=False)
        out = pol.mul(xa, ya)
    elif op == "add":
        out = xa + ya
    elif op == "sub":
        out = xa - ya
    elif op == "mul":
        out = xa * ya
    else:
        raise ShapeError(f"unknown op {op}")
    return NamedTensor(out, canon, pol, bind=False)


# -- pytree: Dynamo flattens to (tensor, static metadata) ---------------------
def _flatten_nt(nt: NamedTensor):
    return [nt.data], (nt.layout, nt.policy)


def _unflatten_nt(children, aux):
    if children[0] is None:
        return None  # tree_map-style ops return None leaves; nothing to rebuild
    layout, policy = aux
    return NamedTensor(children[0], layout, policy, bind=False)


try:
    pytree.register_pytree_node(
        NamedTensor,
        _flatten_nt,
        _unflatten_nt,
        serialized_type_name="sempyt.NamedTensor",
    )
except ValueError:
    # already registered (re-import / reload)
    pass
