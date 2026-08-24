"""Named dimension objects.

A Dim is an identity (the Python object), not a string. Sharing the same Dim
across tensors correlates their axis sizes. ProductDim (H * d) makes merge/split
exact and reversible.
"""

from __future__ import annotations

import contextvars
import inspect
import re
from typing import Iterable, Iterator, Optional, Sequence, Union

from .errors import ShapeError, disagree

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _caller_site() -> Optional[str]:
    """First frame outside sempyt — used as binding provenance."""
    for frame in inspect.stack():
        filename = frame.filename.replace("\\", "/")
        if "/sempyt/" in filename or filename.endswith("/sempyt"):
            continue
        if frame.function.startswith("<"):
            continue
        return f"{filename}:{frame.lineno}"
    return None


class Dim:
    """A named axis. ``size`` is symbolic (None) until bound by a real tensor."""

    __slots__ = ("name", "_size", "role", "_bound_at")

    def __init__(self, name: str, size: Optional[int] = None, role: Optional[str] = None):
        if not _NAME_RE.match(str(name)):
            raise ShapeError(f"invalid dim name {name!r} (need identifier)")
        self.name = str(name)
        self._size = int(size) if size is not None else None
        self.role = role
        self._bound_at = _caller_site() if size is not None else None
        _maybe_register(self)

    @property
    def size(self) -> Optional[int]:
        return self._size

    @size.setter
    def size(self, value: Optional[int]) -> None:
        if value is None:
            self._size = None
            return
        self.bind(int(value))

    @property
    def bound_at(self) -> Optional[str]:
        return self._bound_at

    def bind(self, size: int, site: Optional[str] = None) -> "Dim":
        size = int(size)
        if self._size is None:
            self._size = size
            self._bound_at = site or _caller_site()
        elif self._size != size:
            raise disagree(self, self._size, size)
        return self

    def leaves(self) -> tuple["Dim", ...]:
        return (self,)

    def __mul__(self, other: "Dim") -> "ProductDim":
        if not isinstance(other, Dim):
            return NotImplemented
        return ProductDim(self, other)

    def __rmul__(self, other: "Dim") -> "ProductDim":
        if not isinstance(other, Dim):
            return NotImplemented
        return ProductDim(other, self)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        s = self._size if self._size is not None else "?"
        extra = f", role={self.role}" if self.role else ""
        return f"Dim({self.name!r}, {s}{extra})"

    def __str__(self) -> str:
        s = self._size if self._size is not None else "?"
        return f"{self.name}({s})"


class ProductDim(Dim):
    """A dim that is the product of others: ``HD = H * d``, ``Kf = K * B * H``.

    ``size`` is derived from the factors and is never stored independently.
    """

    __slots__ = ("factors",)

    def __init__(self, *factors: Dim, name: Optional[str] = None):
        flat: list[Dim] = []
        for f in factors:
            if isinstance(f, ProductDim):
                flat.extend(f.factors)
            elif isinstance(f, Dim):
                flat.append(f)
            else:
                raise TypeError(f"ProductDim factors must be Dims, got {type(f)}")
        if len(flat) < 2:
            raise ShapeError("ProductDim needs at least two factors")
        self.factors = tuple(flat)
        derived = None
        if all(f.size is not None for f in self.factors):
            derived = 1
            for f in self.factors:
                derived *= f.size  # type: ignore[operator]
        auto = name or "".join(f.name for f in self.factors)
        # Bypass Dim.__init__ size binding — size is derived.
        if not _NAME_RE.match(auto):
            auto = "prod"
        self.name = auto
        self._size = derived  # kept in sync by the size property
        self.role = "product"
        self._bound_at = None
        _maybe_register(self)

    @property
    def size(self) -> Optional[int]:
        prod = 1
        for f in self.factors:
            if f.size is None:
                return None
            prod *= f.size
        return prod

    @size.setter
    def size(self, value: Optional[int]) -> None:
        if value is None:
            return
        self.bind_product(int(value))

    def bind(self, size: int, site: Optional[str] = None) -> "ProductDim":
        return self.bind_product(int(size), site=site)

    def bind_product(self, size: int, site: Optional[str] = None) -> "ProductDim":
        """Factor ``size`` across leaves; at most one leaf may be unbound."""
        size = int(size)
        known = 1
        unbound: list[Dim] = []
        for leaf in self.leaves():
            if leaf.size is None:
                unbound.append(leaf)
            else:
                if size % leaf.size != 0 and not unbound:
                    pass
                known *= leaf.size
        if unbound:
            if known == 0 or size % known != 0:
                raise ShapeError(
                    f"cannot factor {self.name} size {size} across "
                    f"{[str(f) for f in self.factors]} (known product {known})"
                )
            rest = size // known
            if len(unbound) == 1:
                unbound[0].bind(rest, site=site)
            else:
                raise ShapeError(
                    f"splitting {self} ({size}) into "
                    f"{[u.name for u in unbound]} is under-determined"
                )
        elif self.size != size:
            raise disagree(self, self.size, size)
        return self

    def leaves(self) -> tuple[Dim, ...]:
        out: list[Dim] = []
        for f in self.factors:
            out.extend(f.leaves())
        return tuple(out)

    def __repr__(self) -> str:
        inner = "*".join(f.name for f in self.factors)
        s = self.size if self.size is not None else "?"
        return f"ProductDim({inner}, {s})"


def dims(spec: str, **sizes: int) -> tuple[Dim, ...]:
    """Batch-construct Dim *objects* from a whitespace-separated name string.

    ``dims("B T H d")`` is exactly four ``Dim(...)`` calls. No bytecode
    inspection — names must be written out.
    """
    names = spec.split()
    if not names:
        raise ShapeError("dims() needs at least one name")
    out = []
    for n in names:
        out.append(Dim(n, size=sizes.get(n)))
    return tuple(out)


# ---------------------------------------------------------------------------
# Optional name registry — lets the string form of to() resolve to objects
# ---------------------------------------------------------------------------
_current_registry: contextvars.ContextVar[Optional["DimRegistry"]] = contextvars.ContextVar(
    "sempyt_dim_registry", default=None
)


def _maybe_register(dim: Dim) -> None:
    reg = _current_registry.get()
    if reg is not None:
        reg._put(dim)


class DimRegistry:
    """Maps dim *names* → the Dim *object* they should resolve to.

    Used only by the optional string form ``x.to("B d H T")``. The primary
    API always passes Dim objects.
    """

    def __init__(self, *seed: Dim):
        self._by_name: dict[str, Dim] = {}
        for d in seed:
            self._put(d)

    def _put(self, dim: Dim) -> None:
        existing = self._by_name.get(dim.name)
        if existing is None:
            self._by_name[dim.name] = dim
        elif existing is not dim:
            # Same name, different object: keep the first. String lookup is
            # best-effort; object identity remains the source of truth.
            pass

    def register(self, *ds: Dim) -> "DimRegistry":
        for d in ds:
            self._put(d)
        return self

    def get(self, name: str) -> Dim:
        if name not in self._by_name:
            raise ShapeError(f"DimRegistry has no dim named {name!r}")
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def __enter__(self) -> "DimRegistry":
        self._token = _current_registry.set(self)
        return self

    def __exit__(self, *exc) -> None:
        _current_registry.reset(self._token)

    def __repr__(self) -> str:
        return f"DimRegistry({list(self._by_name)})"


def current_registry() -> Optional[DimRegistry]:
    return _current_registry.get()


_TOKEN_RE = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_]*|\(|\))\s*")


def parse_layout(spec: str, lookup: dict[str, Dim]) -> tuple[Dim, ...]:
    """Parse ``\"B d H T\"`` or ``\"B (H d) T\"`` against a name → Dim map.

    Parentheses build a ProductDim from the grouped objects.
    """
    tokens: list[str] = []
    pos = 0
    s = spec.strip()
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ShapeError(f"cannot parse layout spec {spec!r} at {s[pos:]!r}")
        tokens.append(m.group(1))
        pos = m.end()

    def resolve_name(name: str) -> Dim:
        if name in lookup:
            return lookup[name]
        reg = current_registry()
        if reg is not None and name in reg:
            return reg.get(name)
        raise ShapeError(
            f"unknown dim {name!r} in {spec!r}; pass Dim objects or register the name"
        )

    def parse_seq(i: int) -> tuple[list[Dim], int]:
        items: list[Dim] = []
        while i < len(tokens):
            tok = tokens[i]
            if tok == ")":
                return items, i + 1
            if tok == "(":
                inner, i = parse_seq(i + 1)
                if not inner:
                    raise ShapeError(f"empty group in {spec!r}")
                if len(inner) == 1:
                    items.append(inner[0])
                else:
                    items.append(ProductDim(*inner))
                continue
            items.append(resolve_name(tok))
            i += 1
        return items, i

    items, end = parse_seq(0)
    if end != len(tokens):
        raise ShapeError(f"unbalanced parentheses in {spec!r}")
    return tuple(items)


def lookup_from(*layouts: Sequence[Dim]) -> dict[str, Dim]:
    found: dict[str, Dim] = {}
    for layout in layouts:
        for d in layout:
            found[d.name] = d
            for leaf in d.leaves():
                found[leaf.name] = leaf
    return found


DimLike = Union[Dim, str]
LayoutSpec = Union[str, Sequence[Dim]]
