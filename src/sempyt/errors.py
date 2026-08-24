"""Shape / layout errors with dim names, sizes, and binding provenance."""

from __future__ import annotations


class ShapeError(Exception):
    """A named-axis size or layout mismatch.

    Messages name both offending dims, their sizes, and (when known) the
    file:line where each size was first bound.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _fmt_dim(dim) -> str:
    size = getattr(dim, "size", None)
    site = getattr(dim, "bound_at", None)
    name = getattr(dim, "name", str(dim))
    size_s = "?" if size is None else str(size)
    if site:
        return f"{name}({size_s}) bound at {site}"
    return f"{name}({size_s})"


def disagree(dim, expected, actual, other=None) -> ShapeError:
    left = _fmt_dim(dim)
    if other is not None:
        return ShapeError(
            f"shared dimension sizes disagree: {left} vs {_fmt_dim(other)} "
            f"(expected {expected}, got {actual})"
        )
    return ShapeError(
        f"dimension '{getattr(dim, 'name', dim)}' size disagrees: "
        f"{left}: expected {expected}, got {actual}"
    )


def missing(op: str, dim, side: str, layout) -> ShapeError:
    names = "(" + ", ".join(d.name for d in layout) + ")"
    return ShapeError(f"{op}: dim '{dim.name}' missing from {side} {names}")


def cannot_arrange(old, new, reason: str) -> ShapeError:
    def _ls(layout):
        return "(" + ", ".join(d.name for d in layout) + ")"

    return ShapeError(f"to({_ls(new)}) from {_ls(old)}: {reason}")
