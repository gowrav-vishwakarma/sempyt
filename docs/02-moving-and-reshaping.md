# 02 — Moving and Reshaping

These operations **do no arithmetic**. They only change *where the numbers sit*
— which axis they live on, in what order, grouped how. They are the most common
source of bugs in PyTorch, because PyTorch makes you say *which position* to
move instead of *which axis by name*.

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
