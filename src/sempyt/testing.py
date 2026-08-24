"""Helpers for numeric-equivalence tests."""

from __future__ import annotations

import torch

from .tensor import NamedTensor


def assert_close(a, b, rtol=1e-5, atol=1e-6, msg=""):
    da = a.data if isinstance(a, NamedTensor) else a
    db = b.data if isinstance(b, NamedTensor) else b
    if not torch.allclose(da, db, rtol=rtol, atol=atol, equal_nan=False):
        max_abs = (da - db).abs().max().item()
        raise AssertionError(
            f"{msg} tensors not close: max_abs={max_abs} rtol={rtol} atol={atol}\n"
            f"  a={da.flatten()[:8]}...\n  b={db.flatten()[:8]}..."
        )


def assert_same_object(a, b, msg="expected the identical Python object"):
    if a is not b:
        raise AssertionError(msg)
