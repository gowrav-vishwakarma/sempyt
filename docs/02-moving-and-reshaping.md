# 02 — Moving and Reshaping

These operations **do no arithmetic**. They only change *where the numbers sit*
— which axis they live on, in what order, grouped how. They are the most common
source of bugs in PyTorch, because PyTorch makes you say *which position* to
move instead of *which axis by name*.

---

## The `over` keyword — "which axis to act along"

Before anything else, you need to know one word that appears in almost every
sempyt call: **`over`**.

**What it means:** `over` names the **axis the operation should run along**.
It is the named replacement for PyTorch's `dim=...`.

**Read it as:** "do this operation *along* the `over` axis".

### Example 1 — `split(x, over=qkv_axis, ...)`

A fused QKV projection is one wide axis of size 1536. You want to cut it into
three pieces (q, k, v). Which axis do you cut? The `qkv` one. So you say
`over=qkv_axis`.

| layout | batch | tokens | qkv (1536 wide) |
|--------|-------|--------|-----------------|
| x      | ·     | ·      | q \| k \| v (512 each) |

`split(over=qkv_axis)` → three tensors, each `(batch, tokens, 512)`.

Our example table `x` has 2 rows and 3 columns. The header cell just labels
the axes: read it as "columns c0/c1/c2 across the top, rows r0/r1 down the side".

| x         | c0 | c1 | c2 |
|-----------|----|----|----|
| **r0**    | 1  | 2  | 3  |
| **r1**    | 4  | 5  | 6  |

### Example 2 — `sum(x, over=col)`

You want one total **per row**. So you collapse the `col` axis: add the numbers
*across* each row (`1+2+3` and `4+5+6`).

| result of `sum(x, over=col)` | value |
|------------------------------|-------|
| **r0**                       | 6     |
| **r1**                       | 15    |

The `col` axis no longer exists. The result has just one axis left — `row` (size 2).

### Example 3 — `sum(x, over=row)` (yes, `over` can take `row`)

`over` works for **any** named axis — `row` included. Same table, but now you
want one total **per column**. So you collapse the `row` axis: add the numbers
*down* each column (`1+4`, `2+5`, `3+6`).

| result of `sum(x, over=row)` | value |
|------------------------------|-------|
| **c0**                       | 5     |
| **c1**                       | 7     |
| **c2**                       | 9     |

The `row` axis no longer exists. The result has just one axis left — `col` (size 3).

> **"the axis is gone" — what that really means.** After a sum, the collapsed
> axis is removed from the tensor's list of axes. Example 3's result is a 1-D
> tensor of 3 numbers whose only axis is `col`. Whether you *draw* those 3
> numbers as a row or a column on screen is just presentation — sempyt only
> tracks that there is one axis named `col`, and no `row` axis anymore.

Side-by-side, the only difference is **which axis you name**:

| you want...            | which axis you collapse | call                     | result axis |
|------------------------|-------------------------|--------------------------|-------------|
| one total per row      | `col`                   | `sp.sum(x, over=col)`    | `row` (2)   |
| one total per column   | `row`                   | `sp.sum(x, over=row)`    | `col` (3)   |

```python
import torch
import sempyt as sp
from sempyt import dims, named

row, col = dims("row col", row=2, col=3)
x = named(torch.tensor([[1., 2., 3.], [4., 5., 6.]]), (row, col))

per_row = sp.sum(x, over=col)   # [6, 15]    — one number per row
per_col = sp.sum(x, over=row)   # [5, 7, 9]  — one number per column
```

### The mental rule

> When you see `over=SOMETHING`, ask: *"which axis is this operation acting
> along?"* That axis is the one being chopped, collapsed, summed, scanned,
> or matched up. The other axes are left alone (they are "batch").

| PyTorch | sempyt | meaning |
|---|---|---|
| `x.sum(dim=1)` | `sp.sum(x, over=col)` | collapse the `col` axis |
| `x.split(C, dim=2)` | `sp.split(x, over=qkv, ...)` | cut the `qkv` axis |
| `torch.cat([a,b], dim=0)` | `sp.cat([a,b], over=row)` | glue along the `row` axis |
| `A @ B` | `sp.contract(A, B, over=task)` | sum away the `task` axis |

If you remember one word from this page, remember **`over`**.

---

## `to(...)` — The Universal Rearrange

**What:** Reorder, split, merge, or drop axes in one call.
**When:** Constantly — before a matmul, before a kernel, after a projection.
**Visual:** Swap the two axes (a transpose). The numbers stay the same; the roles
of rows and columns switch.

Before — `x` with layout `(row=2, col=3)`:

| x      | c0 | c1 | c2 |
|--------|----|----|----|
| **r0** | 0  | 1  | 2  |
| **r1** | 3  | 4  | 5  |

After — `x.to(col, row)` with layout `(col=3, row=2)` (rows and columns swapped):

| x.to(col, row) | r0 | r1 |
|----------------|----|----|
| **c0**         | 0  | 3  |
| **c1**         | 1  | 4  |
| **c2**         | 2  | 5  |

**sempyt:** Name the order you want; the library works out the permute.

```python
import torch
from sempyt import dims, named

# 1. Setup your axes and data
row, col = dims("row col", row=2, col=3)
x = named(torch.tensor([[0., 1., 2.], [3., 4., 5.]]), (row, col))

# 2. Transpose by naming the new order
y = x.to(col, row) 
```

---

## ProductDim — The "Shipping Case" Analogy

**What is it?** A `ProductDim` is a "Big Axis" that represents several "Small Axes" fused together.
Think of a factory making widgets:
- You have 8 **Heads** (`H=8`).
- Each head produces a vector of 64 **Features** (`d=64`).
- Total production is 512 numbers.

You can think of this as a `ProductDim(heads, d)`. It's like a **Shipping Case** containing 8 boxes, where each box has 64 widgets. 

**Why use it?**
Most layers (like `nn.Linear`) only understand one "input" dimension. To feed multi-head data into them, you must "box up" (merge) the heads into one big axis of size 512. After the layer, you "unbox" (split) them back.

```python
import torch
from sempyt import dims, named, ProductDim

# Define the axes
heads, d = dims("heads d", heads=8, d=64)

# Create the 'Shipping Case' axis
inner = ProductDim(heads, d) # size is automatically 512

# Create data (8 heads, 64 features each)
x = named(torch.randn(8, 64), (heads, d))

# Box them up: (8, 64) -> (512,)
boxed = x.to(inner) 

# Unbox them perfectly: (512,) -> (8, 64)
unboxed = boxed.to(heads, d) 
```

---

## Stack vs Cat — Folder vs Taller Table

This is the most confusing pair for beginners. Here is how to tell them apart:

### 1. `cat` (Concatenate) — Making a table bigger
**Analogy:** Hiring more workers for the **same shift**. You just add more rows to your existing 2D table. The axis you are adding to *must already exist*.

**Visual (Cat along `row`)** — one taller 2D table, `row=4`:

| row | c0 | c1 | c2 |
|-----|----|----|-----|
| r0  | 0  | 1  | 2   |
| r1  | 3  | 4  | 5   |
| r2  | 10 | 11 | 12  |
| r3  | 13 | 14 | 15  |

(Shift A's rows r0–r1 glued above Shift B's rows — same `col` axis, more rows.)

```python
import torch
import sempyt as sp
from sempyt import dims, named

r, c = dims("row col", row=2, col=3)
shift_a = named(torch.tensor([[0., 1., 2.], [3., 4., 5.]]), (r, c))
shift_b = named(torch.tensor([[10., 11., 12.], [13., 14., 15.]]), (r, c))

# Glue them end-to-end along the 'row' axis
combined = sp.cat([shift_a, shift_b], over=r) # Shape is (row=4, col=3)
```

### 2. `stack` — Creating a new folder
**Analogy:** Taking the Morning Shift table and the Night Shift table and putting them in a **Folder**. You aren't making the tables bigger; you are creating a **brand new axis** called "Shift Type". 

**Visual — two separate tables go *into* one 3-D folder:**

Shift A (`row=2, col=3`) — morning:

| row | c0 | c1 | c2 |
|-----|----|----|-----|
| r0  | 0  | 1  | 2   |
| r1  | 3  | 4  | 5   |

Shift B (`row=2, col=3`) — night:

| row | c0 | c1 | c2 |
|-----|----|----|-----|
| r0  | 10 | 11 | 12  |
| r1  | 13 | 14 | 15  |

**After `stack([A, B], into=shift)` — tree view (shape `(shift=2, row=2, col=3)`):**

Read the first column as the **folder tab**. Each tab holds one full 2×3 table.
The blank cells under a tab mean "still the same shift slice".

| shift   | row | c0 | c1 | c2 |
|---------|-----|----|----|-----|
| shift=0 | r0  | 0  | 1  | 2   |
|         | r1  | 3  | 4  | 5   |
| shift=1 | r0  | 10 | 11 | 12  |
|         | r1  | 13 | 14 | 15  |

Compare to `cat` above: `cat` made **one taller table** (`row=4`). `stack` kept
each table the same size and added a **new axis** (`shift=2`) so you can pick
"which table" with `select(folder, shift, 0)` or `select(folder, shift, 1)`.

```python
import torch
import sempyt as sp
from sempyt import dims, named, Dim

r, c = dims("row col", row=2, col=3)
a = named(torch.tensor([[0., 1., 2.], [3., 4., 5.]]), (r, c))
b = named(torch.tensor([[10., 11., 12.], [13., 14., 15.]]), (r, c))

# Create a NEW axis object
shift = Dim("shift")

# Stack them into the new axis. We put 'shift' at the front (before 'row').
# Result shape: (shift=2, row=2, col=3)
folder = sp.stack([a, b], into=shift, at=r) 
```

---

## 3D example — `batch` folder of 2D tables

A batch of two matrices: layout `(batch=2, row=2, col=3)`. Same tree pattern as
[01 — Foundations](01-foundations.md); here the outer tab is `batch`.

| batch | row | c0 | c1 | c2 |
|-------|-----|----|----|-----|
| b0    | r0  | 1  | 2  | 3   |
|       | r1  | 4  | 5  | 6   |
| b1    | r0  | 10 | 11 | 12  |
|       | r1  | 13 | 14 | 15  |

**`sum(x, over=col)`** — collapse columns inside each batch slice; `col` axis gone:

| batch | value (sum across c0+c1+c2) |
|-------|------------------------------|
| b0    | r0→6, r1→15                  |
| b1    | r0→33, r1→42                 |

Result layout: `(batch=2, row=2)`.

**`x.to(batch, col, row)`** — swap row/col inside every batch tab (a transpose per batch).

---

## Other Useful Moves

### split — Cutting a wide table
Used to split a fused QKV projection. Tiny 3D example `(tokens=2, qkv=6)` → three `(tokens=2, dim=2)` pieces:

| tokens | qkv (wide axis to cut) |
|--------|------------------------|
| t0     | q0 q1 \| k0 k1 \| v0 v1 |
|        |  1  2 \|  3  4 \|  5  6 |
| t1     |  7  8 \|  9 10 \| 11 12 |

After `split(over=qkv, sizes=(2,2,2), names=(q,k,v))`:

| piece | t0 | t1 |
|-------|----|-----|
| **q** | 1, 2 | 7, 8 |
| **k** | 3, 4 | 9, 10 |
| **v** | 5, 6 | 11, 12 |

Each piece keeps `tokens`; the wide `qkv` axis is replaced by a named `q`, `k`, or `v` axis of size 2.

```python
import torch
import sempyt as sp
from sempyt import dims, named, Dim

# Create a packed projection of size 1536
batch, tokens = dims("batch tokens", batch=4, tokens=128)
qkv_axis = Dim("qkv", 1536)
x = named(torch.randn(4, 128, 1536), (batch, tokens, qkv_axis))

# Define the three pieces we want
q_dim, k_dim, v_dim = dims("q k v", q=512, k=512, v=512)

# Cut the 'qkv' axis into three named pieces
q, k, v = sp.split(x, over=qkv_axis, sizes=(512, 512, 512), names=(q_dim, k_dim, v_dim))
```

### select — Picking one item
Used to get the last token of a sequence. From the 3D batch folder above, pick
`batch=b1` and `row=r0` — the `batch` and `row` axes disappear, leaving one row vector:

| col | c0 | c1 | c2 |
|-----|----|----|-----|
|     | 10 | 11 | 12  |

From a `(tokens=3, feat=2)` table, `select(tokens, -1)` picks the last token row:

| feat | f0 | f1 |
|------|----|----|
| t2   | 5  | 6  |

(`tokens` axis gone; result is `(feat=2)`.)

```python
import torch
from sempyt import dims, named

tokens, d = dims("tokens d", tokens=10, d=64)
x = named(torch.randn(10, 64), (tokens, d))

# Pick the last token (index -1). The 'tokens' axis disappears.
last = x[tokens, -1] # Result shape is (64,)
```

---

Next: [03 — Math and Reductions](03-math-and-reductions.md).
