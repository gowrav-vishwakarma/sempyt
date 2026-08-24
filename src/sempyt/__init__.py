"""sempyt — named-dimension tensors over PyTorch."""

from .dim import Dim, DimRegistry, ProductDim, dims
from .errors import ShapeError
from .ops import (
    Index,
    at,
    contract,
    expand_legs,
    gather,
    log_softmax,
    mean,
    outer,
    pack,
    read,
    reduce,
    scan,
    segment,
    softmax,
    sum,
    unsegment,
    unpack,
    write,
)
from .policies import NativeComplex, Real, SplitComplex
from .tensor import NamedTensor, named

__all__ = [
    "Dim",
    "DimRegistry",
    "ProductDim",
    "dims",
    "ShapeError",
    "NamedTensor",
    "named",
    "Real",
    "SplitComplex",
    "NativeComplex",
    "contract",
    "outer",
    "read",
    "write",
    "reduce",
    "sum",
    "mean",
    "softmax",
    "log_softmax",
    "pack",
    "unpack",
    "segment",
    "unsegment",
    "expand_legs",
    "Index",
    "gather",
    "at",
    "scan",
]

__version__ = "0.1.0"
