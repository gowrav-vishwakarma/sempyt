# 01 — Foundations: Tensors as Rosters

Before any code, let's agree on what these grids of numbers *are*. We will use
one running picture: **a factory**.

---

## A number, a list, a grid

**Scalar** — one number. The temperature in the factory today.

```
7
```

**Vector** — a list of numbers, one per thing you track. The output of each of
3 workers today.

```
worker:   w0   w1   w2
         [ 5,  2,  9 ]
```

**Matrix** — a grid: rows × columns. Each **row** is a worker, each **column**
is a station. The number in a cell is "how much worker `w` produced at station `s`".

```
            s0  s1  s2      <- stations (columns)
   w0  [    1   2   3  ]
   w1  [    4   5   6  ]    <- workers (rows)
```

This matrix has **shape** `(2, 3)`: 2 workers, 3 stations.

**Tensor** — the same idea, but with more axes. Think of it as a **roster folder**:
- A 3-D tensor could be `shifts × workers × stations` (one roster matrix per shift).
- A 4-D tensor could be `factories × shifts × workers × stations` (one folder per factory).

---

## Why names matter (The "Who is on axis 1?" problem)

In PyTorch, you refer to axes by their **position**:
- Axis `0` is the first (e.g., `factories`)
- Axis `1` is the second (e.g., `shifts`)
- `-1` is the last (e.g., `stations`)

**The Trap:** If you have a 4-D tensor `(F, S, W, St)`, and you want to sum up all production per worker, you might write `x.sum(dim=2)`.
But wait—did you add a "batch" dimension at the front? Is it now `(B, F, S, W, St)`? If so, `dim=2` is now `shifts`, not `workers`!

You have to constantly track: "Who is at position 2 right now?" This is the source of 90% of silent bugs.

**The sempyt way:** You give each axis a **name** with a `Dim` object.
Instead of `sum(dim=2)`, you say `sum(over=worker)`. It doesn't matter if `worker` is at position 2, 3, or 100—the name stays the same.

---

## The vocabulary

- **Axis** (also called a **dim**): one labeled edge of the grid.
- **Rank** / **ndim**: how many axes. A matrix has rank 2; the 4-D tensor has rank 4.
- **Shape**: the size of each axis, in order. `(2, 3)` above.

---

## Why we batch aligned work

A neural net never processes one worker at a time. It processes a **batch** of
them at once, because the hardware is fastest when it does identical work on a whole grid.
Instead of one matrix, you carry a tensor of shape `(batch, workers, stations)`.

The key word is **aligned**: computation only works if every row means the same
thing (e.g., worker 0 always means worker 0) in every matrix you combine.

---

## Dot product — "line up matched staff, multiply pairs, sum"

The **dot product** of two equal-length vectors lines them up, multiplies the
paired entries, and adds the results. It answers "how much do these two lists
agree?"

```
[1, 2, 3]  ·  [4, 5, 6]
 = 1*4 + 2*5 + 3*6
 = 4 + 10 + 18
 = 32
```

Factory reading: if one vector is "how much of each task a worker did" and the
other is "how much each task is worth", the dot product is that worker's total
value. **Rule: the two vectors must be the same length.**

---

## Matrix multiplication — "every row meets every column"

Multiply matrix `A` (workers × tasks) by matrix `B` (tasks × stations).
The result is matrix `C` (workers × stations).

Each cell of `C` is the **dot product of one row of `A` with one column of `B`**.
The "tasks" axis matches up and is summed away.

```
A (2x3)        B (3x2)          C = A @ B (2x2)
[1 2 3]        [1 0]            [ 1*1+2*0+3*1   1*0+2*1+3*0 ]   [4  2]
[4 5 6]   @    [0 1]      =     [ 4*1+5*0+6*1   4*0+5*1+6*0 ] = [10 5]
               [1 0]
```

**The golden rule:** the touching (inner) sizes must match.
`A` is `(2, 3)` and `B` is `(3, 2)`. The shared `3` (tasks) lines up.
A `(2,3) @ (4,2)` is an error because `3 ≠ 4`.

In sempyt, this is `contract(A, B, over=task)`. You name the axis to line up, and the library handles the rest.

---

Next: [02 — Moving and reshaping](02-moving-and-reshaping.md).
