# Learn tensors with sempyt

This folder teaches you tensors from zero — what a matrix *is*, what each
operation *does to the numbers*, and when you reach for it — then shows the
PyTorch way (and the axis-counting traps) and finally the sempyt one-liner.

You do not need to know linear algebra. Every operation is drawn as a
before → after grid of small numbers, in plain language.

## Reading order

1. [01 — Matrices in plain English](01-matrices-in-plain-english.md)
   What a scalar / vector / matrix / tensor is, using a factory analogy.
   Dot products and matrix multiplication, worked out by hand.

2. [02 — Reshaping and moving](02-reshaping-and-moving.md)
   `to`, reshape, transpose, split, merge, stack, unstack, cat, select, take,
   repeat, shift, roll, flip, pad — moving numbers around without doing math.

3. [03 — Math and reductions](03-math-and-reductions.md)
   Element-wise math, broadcasting, sum/mean/max/softmax, argmax/topk,
   cumsum/cumprod — combining and shrinking numbers.

4. [04 — Tensor algebra](04-tensor-algebra.md)
   contract (matmul), read (dot), outer, gather/at, segment, scan, conj, alias —
   the "match aligned staff" operations, plus a dimension-rules cheatsheet.

5. [05 — Glossary](05-glossary.md)
   Quick, example-first definitions of every term the other pages use.

## The one idea

PyTorch numbers its axes `0, 1, 2, ...` and makes you remember which is which.
sempyt lets you **name** each axis with a `Dim` object (`batch`, `tokens`,
`heads`, ...) and then say *which axis you mean by name* everywhere. That is the
whole trick, and it removes an entire category of silent shape bugs.

When you are comfortable, the terse reference cheatsheet is
[../EXAMPLES.md](../EXAMPLES.md), and the package overview is
[../README.md](../README.md).
