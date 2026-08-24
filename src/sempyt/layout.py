"""Layout = ordered Dims. LayoutPlan lowers name-based rearrange to torch ops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from .dim import Dim, ProductDim, parse_layout, lookup_from
from .errors import ShapeError, cannot_arrange, disagree


class ShapeMap(Mapping[Dim, int]):
    """``tensor.shape`` — keyed by Dim, not integer. ``shape[0]`` is a TypeError."""

    def __init__(self, layout: Sequence[Dim], sizes: Sequence[int]):
        self._layout = tuple(layout)
        self._sizes = tuple(int(s) for s in sizes)
        self._map = {d: s for d, s in zip(self._layout, self._sizes)}

    def __getitem__(self, key):
        if isinstance(key, int):
            raise TypeError(
                "NamedTensor.shape is keyed by Dim, not integer index. "
                "Use tensor.shape[d] or tensor.data.shape."
            )
        if isinstance(key, str):
            for d in self._layout:
                if d.name == key:
                    return self._map[d]
            raise KeyError(key)
        try:
            return self._map[key]
        except KeyError:
            raise KeyError(f"{key} not in layout {[d.name for d in self._layout]}") from None

    def __iter__(self) -> Iterator[Dim]:
        return iter(self._layout)

    def __len__(self) -> int:
        return len(self._layout)

    def __repr__(self) -> str:
        inner = ", ".join(f"{d.name}={s}" for d, s in zip(self._layout, self._sizes))
        return f"ShapeMap({inner})"

    def as_tuple(self) -> tuple[int, ...]:
        return self._sizes


def leaves_of(layout: Sequence[Dim]) -> list[Dim]:
    out: list[Dim] = []
    for d in layout:
        out.extend(d.leaves())
    return out


def bind_layout(layout: Sequence[Dim], shape: Sequence[int], site: Optional[str] = None) -> None:
    if len(layout) != len(shape):
        raise ShapeError(
            f"rank mismatch: tensor has {len(shape)} dims {tuple(shape)}, "
            f"layout has {len(layout)} ({', '.join(d.name for d in layout)})"
        )
    for dim, size in zip(layout, shape):
        if isinstance(dim, ProductDim):
            dim.bind_product(int(size), site=site)
        else:
            dim.bind(int(size), site=site)


def _ids(dims: Sequence[Dim]) -> set[int]:
    return {id(d) for d in dims}


def _consume_by_size(target: Optional[int], candidates: Sequence[Dim]) -> Optional[int]:
    """How many leading candidates multiply to ``target``? None if impossible.

    Binds at most one unbound candidate. Requires ``target`` known, or exactly
    one unbound among a prefix whose other sizes are known.
    """
    if not candidates:
        return None
    # Try prefixes of length 1..len
    for k in range(1, len(candidates) + 1):
        parts = candidates[:k]
        unbound = [p for p in parts if p.size is None]
        known = 1
        ok = True
        for p in parts:
            if p.size is not None:
                known *= p.size
        if target is None:
            # Unbound merge target: only consume the full remaining prefix
            # (all bound). A shorter prefix would be under-determined.
            if len(unbound) == 0 and k == len(candidates):
                return k
            continue
        if unbound:
            if len(unbound) > 1:
                continue
            if known == 0 or target % known != 0:
                continue
            rest = target // known
            unbound[0].bind(rest)
            return k
        if known == target:
            return k
    return None


def _equalize_leaves(
    old_leaves: list[Dim], new_leaves: list[Dim]
) -> tuple[list[Dim], list[Dim], list[int]]:
    """Walk both leaf lists; split/merge by ProductDim identity or size.

    Returns (old_atoms in old order after splits, new_atoms, drop_mask)
    where drop_mask[i] is 1 if that original-expanded slot is a squeezed size-1.
    We instead return old_atoms that will be the reshape of the tensor
    (including size-1 that get squeezed — those are omitted from old_atoms
    and recorded separately).

    Returns:
        expanded_old: Dims of the viewed tensor (after splits, before squeeze)
        squeeze_idx: indices in expanded_old to squeeze (size-1 extras)
        insert_dims: size-1 dims to unsqueeze, appended then permuted
        new_atoms: target leaf order
    """
    # Fast path: same leaf-identity set → just permute
    if _ids(old_leaves) == _ids(new_leaves):
        return list(old_leaves), [], [], list(new_leaves)

    expanded: list[Dim] = []
    squeeze_from_expanded: list[int] = []
    inserts: list[Dim] = []
    target_atoms: list[Dim] = []

    i = j = 0
    old, new = list(old_leaves), list(new_leaves)

    def rem_old():
        return old[i:]

    def rem_new():
        return new[j:]

    while i < len(old) or j < len(new):
        if i < len(old) and j < len(new) and old[i] is new[j]:
            expanded.append(old[i])
            target_atoms.append(new[j])
            i += 1
            j += 1
            continue

        ro, rn = rem_old(), rem_new()
        if ro and rn and _ids(ro) == _ids(rn):
            expanded.extend(ro)
            target_atoms.extend(rn)
            break

        # size-1 drop (old-only)
        if i < len(old) and id(old[i]) not in _ids(rn if rn else []):
            if old[i].size == 1 and id(old[i]) not in _ids(new):
                expanded.append(old[i])
                squeeze_from_expanded.append(len(expanded) - 1)
                i += 1
                continue

        # size-1 insert (new-only)
        if j < len(new) and id(new[j]) not in _ids(ro if ro else []):
            if new[j].size in (None, 1) and id(new[j]) not in _ids(old):
                new[j].bind(1)
                inserts.append(new[j])
                target_atoms.append(new[j])
                j += 1
                continue

        # size-based split: one old dim → k new dims
        if i < len(old) and j < len(new) and id(old[i]) not in _ids(rn):
            k = _consume_by_size(old[i].size, rn)
            if k is not None:
                parts = new[j : j + k]
                prod = 1
                for p in parts:
                    if p.size is None:
                        raise ShapeError(f"factor {p.name} still unbound after split")
                    prod *= p.size
                if old[i].size is not None and prod != old[i].size:
                    raise cannot_arrange(
                        old_leaves,
                        new_leaves,
                        f"factors {[p.name for p in parts]} multiply to {prod}, "
                        f"but {old[i].name} is {old[i].size}",
                    )
                if old[i].size is None:
                    old[i].bind(prod)
                expanded.extend(parts)
                target_atoms.extend(parts)
                i += 1
                j += k
                continue

        # size-based merge: k old dims → one new dim
        if i < len(old) and j < len(new) and id(new[j]) not in _ids(ro):
            k = _consume_by_size(new[j].size, ro)
            if k is not None:
                parts = old[i : i + k]
                prod = 1
                for p in parts:
                    prod *= p.size  # type: ignore[operator]
                if new[j].size is None:
                    new[j].bind(prod)
                elif new[j].size != prod:
                    raise disagree(new[j], new[j].size, prod)
                expanded.extend(parts)
                target_atoms.extend(parts)
                i += k
                j += 1
                continue

        raise cannot_arrange(
            old_leaves,
            new_leaves,
            f"cannot match remaining old {[d.name for d in ro]} "
            f"to new {[d.name for d in rn]}",
        )

    return expanded, squeeze_from_expanded, inserts, target_atoms


def _need_size(d: Dim, what: str) -> int:
    if d.size is None:
        raise ShapeError(f"{what}: dim {d.name} is unbound")
    return d.size


@dataclass(frozen=True)
class LayoutPlan:
    """Cached lowering of one named rearrange to permute / view / squeeze / unsqueeze."""

    expand_shape: tuple[int, ...]
    squeeze: tuple[int, ...]
    insert_at: tuple[int, ...]  # unsqueeze positions in the *post-permute* tensor... see apply
    perm: tuple[int, ...]
    n_inserts: int
    final_shape: tuple[int, ...]
    identity: bool

    def apply(self, data):
        if self.identity:
            return data
        x = data
        if x.shape != self.expand_shape:
            x = x.reshape(self.expand_shape)
        if self.squeeze:
            for idx in reversed(self.squeeze):
                x = x.squeeze(idx)
        if self.n_inserts:
            # size-1 dims were conceptually appended; unsqueeze at end
            for _ in range(self.n_inserts):
                x = x.unsqueeze(-1)
        if self.perm != tuple(range(x.ndim)):
            x = x.permute(self.perm)
        if x.shape != self.final_shape:
            x = x.reshape(self.final_shape)
        return x


_PLAN_CACHE: dict[tuple, LayoutPlan] = {}
_PLAN_CACHE_MAX = 2048


def _cache_key(old: Sequence[Dim], new: Sequence[Dim], shape: Sequence[int]) -> tuple:
    return (
        tuple(id(d) for d in old),
        tuple(id(d) for d in new),
        tuple(int(s) for s in shape),
        tuple(d.size for d in leaves_of(old)),
        tuple(d.size for d in leaves_of(new)),
    )


def plan_to(
    old_layout: Sequence[Dim],
    new_layout: Sequence[Dim],
    tensor_shape: Sequence[int],
) -> LayoutPlan:
    """Build (or fetch) the LayoutPlan taking ``old_layout`` to ``new_layout``."""
    key = _cache_key(old_layout, new_layout, tensor_shape)
    cached = _PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    bind_layout(old_layout, tensor_shape)

    # No-op: same objects, same order
    if len(old_layout) == len(new_layout) and all(a is b for a, b in zip(old_layout, new_layout)):
        # still bind new (same objects)
        plan = LayoutPlan(
            expand_shape=tuple(int(s) for s in tensor_shape),
            squeeze=(),
            insert_at=(),
            perm=tuple(range(len(old_layout))),
            n_inserts=0,
            final_shape=tuple(int(s) for s in tensor_shape),
            identity=True,
        )
        _store_plan(key, plan)
        return plan

    old_leaves = leaves_of(old_layout)
    new_leaves = leaves_of(new_layout)

    # Bind product factors from already-bound parents
    for d, size in zip(old_layout, tensor_shape):
        if isinstance(d, ProductDim):
            d.bind_product(int(size))

    expanded, squeeze, inserts, new_atoms = _equalize_leaves(old_leaves, new_leaves)

    # expanded is in "view of data" order (old splits applied, size-1 extras kept)
    expand_shape = []
    # When we split an old dim, expanded contains the *new* parts; their sizes
    # must multiply to the original axis. Reconstruct expand_shape from
    # expanded dims, but squeeze targets are size-1 extras sitting in expanded.
    for d in expanded:
        expand_shape.append(_need_size(d, "to()"))

    # After squeeze, remaining dims:
    kept = [d for idx, d in enumerate(expanded) if idx not in squeeze]
    # Append inserts (size 1)
    current = list(kept) + list(inserts)
    n_inserts = len(inserts)

    # Permute current → new_atoms. new_atoms is flattened new_layout.
    # After inserts, current's identity set must equal new_atoms.
    if _ids(current) != _ids(new_atoms):
        # new_atoms from equalize is flatten(new_layout); inserts are already
        # in new_layout so they appear in new_atoms. current should match.
        raise cannot_arrange(
            old_layout,
            new_layout,
            f"internal: current {[d.name for d in current]} "
            f"!= target {[d.name for d in new_atoms]}",
        )

    pos_of: dict[int, list[int]] = {}
    for i, d in enumerate(current):
        pos_of.setdefault(id(d), []).append(i)
    perm = []
    used: dict[int, int] = {}
    for d in new_atoms:
        k = id(d)
        slot = used.get(k, 0)
        perm.append(pos_of[k][slot])
        used[k] = slot + 1

    final_shape = []
    for d in new_layout:
        if d.size is None:
            # derive from leaves
            prod = 1
            for leaf in d.leaves():
                prod *= _need_size(leaf, "to()")
            if isinstance(d, ProductDim):
                pass
            else:
                d.bind(prod)
        final_shape.append(_need_size(d, "to()"))

    identity = (
        not squeeze
        and n_inserts == 0
        and perm == list(range(len(perm)))
        and tuple(expand_shape) == tuple(final_shape)
        and tuple(tensor_shape) == tuple(final_shape)
    )

    plan = LayoutPlan(
        expand_shape=tuple(expand_shape),
        squeeze=tuple(squeeze),
        insert_at=(),
        perm=tuple(perm),
        n_inserts=n_inserts,
        final_shape=tuple(final_shape),
        identity=identity,
    )
    _store_plan(key, plan)
    return plan


def _store_plan(key, plan: LayoutPlan) -> None:
    if len(_PLAN_CACHE) >= _PLAN_CACHE_MAX:
        _PLAN_CACHE.clear()
    _PLAN_CACHE[key] = plan


def resolve_spec(spec, source_layout: Sequence[Dim]) -> tuple[Dim, ...]:
    """Coerce a to()-spec (objects, a sequence, or a string) to a Dim tuple."""
    if isinstance(spec, str):
        return parse_layout(spec, lookup_from(source_layout))
    if isinstance(spec, Dim):
        return (spec,)
    return tuple(spec)


def layout_str(layout: Sequence[Dim]) -> str:
    return "(" + ", ".join(d.name for d in layout) + ")"
