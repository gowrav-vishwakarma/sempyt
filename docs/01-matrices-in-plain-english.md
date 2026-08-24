# 01 — Matrices in plain English

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

**Tensor** — the same idea, more axes. A 3-D tensor could be
`shifts × workers × stations` (one matrix per shift). A 4-D tensor could be
`factories × shifts × workers × stations`. Nothing new — just more labels on the
edges.

---

## The vocabulary, once

- **Axis** (also called a **dim**): one labeled edge of the grid. The matrix
  above has two axes: `worker` (size 2) and `station` (size 3).
- **Rank** / **ndim**: how many axes. A matrix has rank 2; the 4-D tensor has rank 4.
- **Shape**: the size of each axis, in order. `(2, 3)` above.

In PyTorch you refer to an axis by **position**: axis `0` is the first, axis `1`
is the second, and `-1` is the last. That is the whole source of the confusion
this library removes — more on that from page 02 onward.

---

## Why we batch aligned work

A neural net never processes one worker at a time. It processes a **batch** of
them at once, because the arithmetic is identical for each and the hardware is
fastest when it does the same operation on a whole grid. So instead of one
vector of shape `(3,)`, you carry a matrix of shape `(batch, 3)` — the same
computation, done for every row in parallel.

The key word is **aligned**: a batch only works if every row means the same
thing (row = a worker, column = a station) in every matrix you combine. Keeping
axes aligned is the entire job of the next three pages.

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
value. **Rule: the two vectors must be the same length** — you cannot pair up
mismatched lists.

---

## Matrix multiplication — "every row meets every column"

Multiply a matrix `A` (workers × tasks) by a matrix `B` (tasks × stations) and
you get a matrix `C` (workers × stations). Each cell of `C` is the **dot
product of one row of `A` with one column of `B`** — the tasks "cancel out"
because they are matched up.

```
A (2x3)        B (3x2)          C = A @ B (2x2)
[1 2 3]        [1 0]            [ 1*1+2*0+3*1   1*0+2*1+3*0 ]   [4  2]
[4 5 6]   @    [0 1]      =     [ 4*1+5*0+6*1   4*0+5*1+6*0 ] = [10 5]
               [1 0]
```

Read the top-left cell: worker `w0`'s row `[1,2,3]` dotted with station `s0`'s
column `[1,0,1]` = `1+0+3 = 4`.

**The one rule that matters:** the touching (inner) sizes must match.
`A` is `(2, 3)` and `B` is `(3, 2)` — the shared `3` (tasks) lines up, and the
result keeps the outer sizes `(2, 2)`. A `(2,3) @ (4,2)` is an error, because
`3 ≠ 4`: there is no way to pair up the middle.

In sempyt this is `contract(A, B, over=tasks)` — you name the axis that should
line up and disappear, and the library checks the sizes for you. That is page 04.

---

## What you now know

- A tensor is a labeled grid; each edge is an **axis** with a **size**.
- We batch because the same math applies to every row.
- A **dot product** pairs up two equal lists and sums the products.
- A **matmul** is "every row of the left dotted with every column of the right",
  and it only works when the touching sizes match.

Next: [02 — Reshaping and moving](02-reshaping-and-moving.md).
