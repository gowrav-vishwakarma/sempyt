# Learn Tensors with sempyt

This folder teaches you tensors from zero — what a matrix *is*, what each
operation *does to the numbers*, and when you reach for it — then shows the
PyTorch way (and the axis-counting traps) and finally the sempyt one-liner.

You do not need to know linear algebra. Every operation is explained with
visual before → after grids and plain-language analogies.

## Reading Order

1. [01 — Foundations: Tensors as Rosters](01-foundations.md)
   What a scalar / vector / matrix / tensor is. Dot products and matrix multiplication explained with a factory analogy.

2. [02 — Moving and Reshaping](02-moving-and-reshaping.md)
   `to`, `stack` vs `cat`, `ProductDim` (the "Shipping Case"), split, select, and other ways to move numbers without doing math.

3. [03 — Math and Reductions](03-math-and-reductions.md)
   Element-wise math, broadcasting ("stretching"), sum/mean/max/softmax, and finding winners with argmax/topk.

4. [04 — Tensor Algebra](04-tensor-algebra.md)
   `contract` (matmul by name), Associative memory (`read` / `outer`), folding time with `segment`, and the "Running Total" (`scan`).

5. [05 — Glossary](05-glossary.md)
   Quick, example-first definitions of terms like rank, shape, and broadcasting.

## The one idea

PyTorch numbers its axes `0, 1, 2, ...` and makes you remember which is which.
sempyt lets you **name** each axis with a `Dim` object (`batch`, `tokens`,
`heads`, ...) and then say *which axis you mean by name* everywhere. That is the
whole trick, and it removes an entire category of silent shape bugs.

When you are comfortable, the terse reference cheatsheet is
[../EXAMPLES.md](../EXAMPLES.md), and the package overview is
[../README.md](../README.md).
