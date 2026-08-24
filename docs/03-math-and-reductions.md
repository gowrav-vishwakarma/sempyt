# 03 — Math and Reductions

Page 02 moved numbers around. This page **combines** them: element-wise math,
broadcasting, and the reductions that shrink an axis down to a single number.

---

## Element-wise math — same shape in, same shape out

**What:** Apply an operation to every cell independently. The grid keeps its exact shape.
**When:** Activations (silu/gelu), scaling, gating.

```python
import torch
import sempyt as sp
from sempyt import dims, named

r, c = dims("row col", row=2, col=2)
x = named(torch.tensor([[1., 2.], [3., 4.]]), (r, c))

# All these keep the shape (row=2, col=2)
y = sp.square(x)   # [[1, 4], [9, 16]]
z = sp.sigmoid(x)  # probabilities
w = x * 2.0        # scale every cell
```

---

## Broadcasting — "Stretching" to Match

**What:** When you add a small vector to a large matrix, PyTorch "stretches" the vector so it matches every row or column. 

### The PyTorch Trap
PyTorch aligns shapes on the **right**. If you have `(batch, time, features)` and a `(batch, features)` bias, you often have to write `bias.unsqueeze(1)` to make the axes line up positionally.

### The sempyt Solution: Name Alignment
In sempyt, broadcasting happens over the **union of named axes**. If both tensors share a `Dim` object, they line up automatically, no matter where they are in the shape.

**Visual (Stretching `gain` over `x`):**

| gain (`col=3`) | x (`row=2, col=3`) | Result (`x * gain`) |
| :--- | :--- | :--- |
| `[10, 100, 1000]` | `[1, 2, 3]` <br> `[4, 5, 6]` | `[10, 200, 3000]` <br> `[40, 500, 6000]` |
| (stretches down) | (stays same) | (multiplied vertically) |

```python
import torch
from sempyt import dims, named

row, col = dims("row col", row=2, col=3)
x = named(torch.tensor([[1., 2., 3.], [4., 5., 6.]]), (row, col))

# This gain only has the 'col' axis. 
# sempyt stretches it across 'row' automatically because the names match.
gain = named(torch.tensor([10., 100., 1000.]), (col,))

y = x * gain # No unsqueeze needed!
```

---

## Reductions — "Collapsing" an Axis

**What:** Shrink an axis by combining all its numbers (summing them, averaging them).
**When:** Computing total loss, attention scores, or feature norms.

**Visual (Summing over `col`):**

```
x (row=2, col=3)          sum(over=col)           Result (row=2)
  c0 c1 c2
r0 [1 2 3]      --->      1 + 2 + 3      --->     r0 [ 6 ]
r1 [4 5 6]      --->      4 + 5 + 6      --->     r1 [15 ]
                                                  (col axis is gone)
```

```python
import torch
import sempyt as sp
from sempyt import dims, named

r, c = dims("row col", row=2, col=3)
x = named(torch.tensor([[1., 2., 3.], [4., 5., 6.]]), (r, c))

# Collapse the 'col' axis to get a total for every row
total = sp.sum(x, over=c) # Result layout: (row,)
```

---

## Finding the Winners (argmax / topk)

### 1. argmax — "Which worker is best?"
Returns the **index** (the position) of the largest value. 

```python
import torch
import sempyt as sp
from sempyt import dims, named

v = dims("vocab", vocab=1000)
logits = named(torch.randn(1000), (v,))

# Find which word ID is most likely
idx = sp.argmax(logits, over=v) 
print(idx.values) # e.g., 42
```

### 2. topk — "Who are the top 2?"
Keeps the `k` largest values and their indices, creating a **new named axis** for the winners.

**Visual (topk over `col`, k=2):**

| Input `x` (col=4) | `topk(over=col, k=2, kdim=K)` |
| :--- | :--- |
| `[1, 9, 2, 8]` | `vals: [9, 8]` <br> `ids: [1, 3]` |
| | (Result axis is `K`) |

```python
import torch
import sempyt as sp
from sempyt import dims, named, Dim

c = Dim("col", 4)
x = named(torch.tensor([1., 9., 2., 8.]), (c,))

# We want the top 2 winners on a new axis 'K'
K = Dim("topk_winners", 2)
vals, ids = sp.topk(x, over=c, k=2, kdim=K)
```

---

## Running Totals (cumsum)

**What:** Each step along the axis becomes the sum of "everything seen so far".
**When:** Causal masks, probability sums.

```
Input x (col=4)        [ 1,   2,   3,   4  ]
cumsum(over=col)  ---> [ 1, 1+2, 3+3, 6+4 ]  --->  [ 1, 3, 6, 10 ]
```

```python
import torch
import sempyt as sp
from sempyt import dims, named

c = Dim("col", 4)
x = named(torch.tensor([1., 2., 3., 4.]), (c,))

y = sp.cumsum(x, over=c) # Result: [1, 3, 6, 10]
```

---

Next: [04 — Tensor Algebra](04-tensor-algebra.md).
