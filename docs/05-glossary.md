# 05 — Glossary

Quick, example-first definitions of every term the other pages use. Each entry
is one tiny grid.

---

**Scalar** — a single number, rank 0. `7`

**Vector** — one axis, rank 1. `[5, 2, 9]` (shape `(3,)`)

**Matrix** — two axes, rank 2. `[[1,2,3],[4,5,6]]` (shape `(2,3)`)

**Tensor** — the general case: any number of axes. A scalar, vector, and matrix
are all tensors.

**Axis / dim** — one labeled edge of a tensor. The matrix above has axis `0`
(rows, size 2) and axis `1` (columns, size 3). In sempyt each axis is a named
`Dim` object instead of a position.

**Rank / ndim** — how many axes a tensor has. A matrix has rank 2.

**Shape** — the size of each axis, in order. `[[1,2,3],[4,5,6]]` has shape
`(2, 3)`. In sempyt you ask `x.shape[some_dim]` by name, never `x.shape[0]`.

**Element** — one number in the grid. The `5` above is the element at
row 1, column 1.

**Element-wise** — an operation applied to each cell independently, keeping the
shape. `square([[1,2],[3,4]]) = [[1,4],[9,16]]`.

**Batch dimension** — the leading axis that holds many independent examples so
the same math runs on all of them at once. `(batch=32, tokens, channels)`.

**Dot product** — line up two equal-length vectors, multiply the pairs, sum:
`[1,2,3]·[4,5,6] = 32`. Requires equal lengths.

**Inner product** — the generalization of a dot product to tensors: contract one
shared axis. In sempyt this is `read(a, b, over=axis)`.

**Outer product** — the opposite: combine two vectors into a grid where cell
`(i,j)` is `a[i]*b[j]`. `[1,2] ⊗ [1,0,1] = [[1,0,1],[2,0,2]]`. In sempyt,
`outer(a, b, over=(di, dj))`.

**Matmul** — matrix multiplication: each output cell is the dot product of a row
of the left with a column of the right. The touching (inner) sizes must match.
In sempyt, `contract(a, b, over=shared)`.

**Broadcasting** — stretching a smaller tensor to match a bigger one so an
element-wise op can proceed. PyTorch aligns shapes on the right and stretches
any size-1 axis. sempyt broadcasts over the union of named axes, so you never
`unsqueeze` by hand. (Full treatment in
[03 — Math and reductions](03-math-and-reductions.md#broadcasting--stretch-a-smaller-grid-to-match-a-bigger-one).)

**keepdim** — a PyTorch flag on reductions. `x.sum(dim=1)` drops axis 1;
`x.sum(dim=1, keepdim=True)` keeps it as size 1 so a later broadcast still
lines up positionally. sempyt drops the named axis and broadcasts by name, so
you never need it.

**Contiguous / stride** — how the numbers are laid out in memory. Some ops
(transpose) only *relabel* the layout without moving data; a later `view` then
fails until you call `.contiguous()` to actually reorder the bytes. In sempyt
you call `.to(...)` to rearrange and `.raw(...)` to get a contiguous plain
tensor for a kernel, so this never surfaces as an error.

**Index** — a sempyt integer tensor that remembers which axis it points into.
Produced by `argmax`/`topk`, consumed by `gather`/`at`.

**Dim / ProductDim** — sempyt's axis objects. A `Dim` is one named axis; a
`ProductDim` is a fused axis that is the product of others (so split and merge
are exact inverses).

**Layout** — the ordered tuple of `Dim`s describing a `NamedTensor`'s axes.

**Policy** — how element-wise math behaves on a `NamedTensor`: `Real`,
`SplitComplex` (complex stored as `[real, imag]` on a `pair` axis), or
`NativeComplex`.

---

Back to the [index](README.md), or the terse [reference cheatsheet](../EXAMPLES.md).
