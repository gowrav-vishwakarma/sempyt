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

```
x  (batch, tokens, qkv=1536)
                qkv axis
                 ↓
batch   tokens  [ q | k | v ]      split(over=qkv_axis)
                 |-------------|            -->
                                          q (batch, tokens, 512)
                                          k (batch, tokens, 512)
                                          v (batch, tokens, 512)
              (the 'qkv' axis is the one being chopped)
```

### Example 2 — `sum(x, over=col)`

You have a `(row, col)` table and want one total per row. Which axis do you
collapse? The `col` one. So you say `over=col`.

```
x  (row, col)
   c0 c1 c2
r0 [1  2  3]   sum(over=col)   -->   r0 [ 6 ]     (col is gone)
r1 [4  5  6]                        r1 [15 ]
   (collapse along the 'col' axis)
```

### Example 3 — `sum(x, over=row)` (yes, `over` can take `row`)

`over` works for **any** named axis — `row` included. Same table, but now you
want one total per **column** (collapse the rows instead). Which axis do you
collapse? The `row` one. So you say `over=row`.

```
x  (row, col)
   c0 c1 c2
r0 [1  2  3]   sum(over=row)   -->   c0 [ 5 ]    (row is gone)
r1 [4  5  6]                        c1 [ 7 ]
                                       c2 [ 9 ]
   (collapse along the 'row' axis)
```

Side-by-side, the only difference is **which axis you name**:

| you want... | which axis disappears | call |
|---|---|---|
| one total per row | `col` | `sp.sum(x, over=col)` |
| one total per column | `row` | `sp.sum(x, over=row)` |

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
**Visual:** Swap the two axes (a transpose).

```
x (row=2, col=3)          x.to(col, row)  ->  (col=3, row=2)
   c0 c1 c2                    r0  r1
r0 [0 1 2]                 c0 [ 0   3 ]
r1 [3 4 5]                 c1 [ 1   4 ]
                           c2 [ 2   5 ]
```

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

**Visual (Cat along `row`):**

| Shift A (`row=2, col=3`) | Shift B (`row=2, col=3`) | `cat([A, B], over=row)` |
| :--- | :--- | :--- |
| `[0, 1, 2]` <br> `[3, 4, 5]` | `[10, 11, 12]` <br> `[13, 14, 15]` | `[0, 1, 2]` <br> `[3, 4, 5]` <br> `[10, 11, 12]` <br> `[13, 14, 15]` |
| | | (Row size becomes 4) |

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

**Visual (Stack into `shift`):**

```
Shift A (row, col)        Shift B (row, col)         stack([A, B], into=shift)
[0 1 2]                   [10 11 12]                 shift=0: [0 1 2] [3 4 5]
[3 4 5]                   [13 14 15]        --->     shift=1: [10 11 12] [13 14 15]
                                                     (Result is 3D)
```

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

## Other Useful Moves

### split — Cutting a wide table
Used to split a fused QKV projection.

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
Used to get the last token of a sequence.

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
