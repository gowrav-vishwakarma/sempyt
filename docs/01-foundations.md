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

### How we draw 2D / 3D / 4D in these docs

Higher dimensions are hard to picture as one flat grid. We use a **tree table**:
the left column(s) are **folder tabs** (outer axes). Blank cells under a tab mean
"still inside that slice". The right columns are the inner 2D table.

**2D** — one roster matrix `(worker=2, station=3)`:

| worker | s0 | s1 | s2 |
|--------|----|----|-----|
| w0     | 1  | 2  | 3   |
| w1     | 4  | 5  | 6   |

**3D** — a folder of two rosters `(shift=2, worker=2, station=3)`:

| shift   | worker | s0 | s1 | s2 |
|---------|--------|----|----|-----|
| shift=0 | w0     | 1  | 2  | 3   |
|         | w1     | 4  | 5  | 6   |
| shift=1 | w0     | 10 | 11 | 12  |
|         | w1     | 13 | 14 | 15  |

**4D** — two factories, each with the same shift folder `(factory=2, shift=2, worker=2, station=2)`:

| factory | shift   | worker | s0 | s1 |
|---------|---------|--------|----|-----|
| f0      | shift=0 | w0     | 1  | 2   |
|         |         | w1     | 3  | 4   |
|         | shift=1 | w0     | 5  | 6   |
|         |         | w1     | 7  | 8   |
| f1      | shift=0 | w0     | 9  | 10  |
|         |         | w1     | 11 | 12  |
|         | shift=1 | w0     | 13 | 14  |
|         |         | w1     | 15 | 16  |

Read it inside-out: **station** columns are the cells, **worker** rows are inside
each shift tab, **shift** tabs sit inside each **factory** tab. Rank 4 = four
nested labels before you reach a single number.

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

| A (`worker×task`) | | t0 | t1 | t2 |
|-------------------|---|----|----|-----|
| | w0 | 1 | 2 | 3 |
| | w1 | 4 | 5 | 6 |

| B (`task×station`) | | s0 | s1 |
|--------------------|---|----|-----|
| | t0 | 1 | 0 |
| | t1 | 0 | 1 |
| | t2 | 1 | 0 |

| C = `contract(A, B, over=task)` (`worker×station`) | | s0 | s1 |
|------------------------------------------------------|---|----|-----|
| | w0 | 4 | 2 |
| | w1 | 10 | 5 |

Each cell is one row of A dotted with one column of B; the shared `task` axis is summed away.

**The golden rule:** the touching (inner) sizes must match.
`A` is `(2, 3)` and `B` is `(3, 2)`. The shared `3` (tasks) lines up.
A `(2,3) @ (4,2)` is an error because `3 ≠ 4`.

In sempyt, this is `contract(A, B, over=task)`. You name the axis to line up, and the library handles the rest.

---

Next: [02 — Moving and reshaping](02-moving-and-reshaping.md).
