# 03 — Math and reductions

Page 02 moved numbers around. This page **combines** them: element-wise math,
broadcasting, and the reductions that shrink an axis down to a single number.

Setup:

```python
import torch
import sempyt as sp
from sempyt import Dim, dims, named

row, col = dims("row col", row=2, col=3)
x = named(torch.tensor([[1., 5., 2.],
                        [4., 0., 6.]]), (row, col))
```

```
x  (row=2, col=3)
        c0  c1  c2
   r0 [ 1   5   2 ]
   r1 [ 4   0   6 ]
```

---

## Element-wise math — same shape in, same shape out

**What:** apply an operation to every cell independently. The grid keeps its
exact shape and axes.

```python
sp.square(x)      # [[1, 25, 4], [16, 0, 36]]
sp.exp(x); sp.log(x); sp.sqrt(x); sp.sigmoid(x); sp.silu(x); sp.gelu(x)
sp.clamp(x, min=0, max=4)   # [[1, 4, 2], [4, 0, 4]]
```

**When:** activations (silu/gelu/sigmoid), gating, scaling. These are the bulk
of a neural net between the matmuls.

---

## Broadcasting — stretch a smaller grid to match a bigger one

**What:** when two operands have different shapes, PyTorch stretches the smaller
one (without copying memory) so the operation can proceed cell-by-cell.

**The PyTorch rule (two parts):** align shapes on the **right**, then each axis
must either match or be size `1` (which stretches).

**Worked — a per-column gain `(col,)` times the matrix `(row, col)`:**

```
gain (col,)        x (row, col)            x * gain
[10 100 1000]      [1 5 2]      ---->      [ 10  500 2000 ]
                   [4 0 6]                 [ 40    0 6000 ]
                   (gain stretched down every row)
```

**Worked — a per-row gain `(row, 1)`:**

```
gain (row,1)       x (row, col)            x * gain
[10]               [1 5 2]      ---->      [10 50 20]
[100]              [4 0 6]                 [400 0 600]
                   (gain stretched across every column)
```

**A shape that FAILS:** `(2,3) + (2,)` — right-aligned, the trailing `3` meets
`2`, neither is `1`, so PyTorch raises an error. There is no valid stretch.

**The PyTorch gotcha:** you often have to insert size-1 axes by hand with
`unsqueeze` so the right-alignment lands where you intend. In 4-D that is
`gain.unsqueeze(-1).unsqueeze(-1)` and a lot of counting.

**sempyt:** broadcasting is over the **union of named axes** — you never
unsqueeze. A `(col,)` gain times a `(row, col)` matrix just works, because both
sides agree the shared axis is `col`:

```python
gain = named(torch.tensor([10., 100., 1000.]), (col,))
x * gain          # (row, col) — no unsqueeze, no counting
```

---

## reduce / sum / mean / max — collapse an axis to one number

**What:** shrink an axis by combining its cells (add them, average them, take
the largest). That axis disappears from the result.

**Visual (sum over `col`):**

```
x (row, col)              sum(x, over=col)  ->  (row,)
[1 5 2]                   r0: 1+5+2 = 8
[4 0 6]                   r1: 4+0+6 = 10
```

```python
sp.sum(x, over=col)              # [8, 10]        — axis 'col' is gone
sp.mean(x, over=col)             # [2.67, 3.33]
sp.reduce(x, over=col, how="max")  # [5, 6]
sp.reduce(x, over=(row, col), how="sum")  # 18 — collapse several axes
```

**When:** losses (mean over batch), attention weights, pooling, normalization.

**The `keepdim` gotcha:** PyTorch's `x.sum(dim=1)` *drops* the axis, so a later
broadcast against the original shape fails unless you pass `keepdim=True`.
sempyt reductions always drop the named axis, and because broadcasting is by
name (not position) you never need `keepdim` — a `(row,)` result still
broadcasts cleanly against a `(row, col)` tensor.

---

## softmax — turn a row of scores into probabilities

**What:** exponentiate each cell along an axis, then divide by the total, so the
cells along that axis sum to 1.

**Visual (softmax over `col`, row r0):**

```
[1, 5, 2]  ->  exp -> [2.7, 148.4, 7.4]  ->  /158.5  ->  [0.017, 0.936, 0.047]
```

**When:** the final layer over the vocabulary axis; attention over the key axis.

```python
sp.softmax(x, over=col)     # each row now sums to 1
```

**PyTorch:** `torch.softmax(x, dim=1)`. **sempyt:** name the axis.

---

## argmax / argmin — which position wins

**What:** instead of the winning *value*, return the winning *index* along an
axis. The axis is replaced by an `Index` (integers that point into it).

**Visual (argmax over `col`):**

```
x (row, col)              argmax(x, over=col)  ->  Index over 'col'
[1 5 2]                   r0: position 1 (the 5)
[4 0 6]                   r1: position 2 (the 6)
```

```python
idx = sp.argmax(x, over=col)   # idx.values == [1, 2]; idx.over is 'col'
```

**When:** greedy decoding (argmax over the vocab axis), nearest-neighbour
cleanup. The result feeds straight into `gather`/`at` (page 04).

---

## topk — the k best, on a new axis

**What:** like argmax but keep the top `k` values *and* their indices, placed on
a fresh axis of size `k`.

**Visual (topk over `col`, k=2):**

```
x (row, col)              topk(x, over=col, k=2, kdim=K)
[1 5 2]                   vals: [5, 2]    ids: [1, 2]
[4 0 6]                   vals: [6, 4]    ids: [2, 0]
```

```python
K = Dim("K")
vals, ids = sp.topk(x, over=col, k=2, kdim=K)
```

**When:** top-k sampling, top-k MoE routing (keep the k best experts per token).

---

## cumsum / cumprod — running totals along an axis

**What:** each cell becomes the sum (or product) of everything up to and
including it along the axis. Shape is preserved.

**Visual (cumsum over `col`):**

```
x (row, col)              cumsum(x, over=col)
[1 5 2]                   [1  6  8]      (1, 1+5, 1+5+2)
[4 0 6]                   [4  4 10]
```

**When:** cumulative attention, running normalizers, and inside linear-recurrence
(`scan`) implementations.

```python
sp.cumsum(x, over=col); sp.cumprod(x, over=col)
```

---

Next: [04 — Tensor algebra](04-tensor-algebra.md).
