# 02 — Reshaping and moving

These operations **do no arithmetic**. They only change *where the numbers sit*
— which axis they live on, in what order, grouped how. They are the most common
source of bugs in PyTorch, because PyTorch makes you say *which position* to
move instead of *which axis by name*.

Setup for every example below:

```python
import torch
import sempyt as sp
from sempyt import Dim, dims, named

row, col = dims("row col", row=2, col=3)
x = named(torch.tensor([[0., 1., 2.],
                        [3., 4., 5.]]), (row, col))
```

```
x  (row=2, col=3)
        c0  c1  c2
   r0 [ 0   1   2 ]
   r1 [ 3   4   5 ]
```

---

## `to(...)` — the universal rearrange

**What:** reorder, split, merge, or drop axes in one call.
**When:** constantly — before a matmul, before a kernel, after a projection.
**Visual:** swap the two axes (a transpose):

```
x (row, col)              x.to(col, row)  ->  (col, row)
   c0 c1 c2                    r0  r1
r0 [0 1 2]                 c0 [ 0   3 ]
r1 [3 4 5]                 c1 [ 1   4 ]
                           c2 [ 2   5 ]
```

**PyTorch:** `x.t()` for 2-D, or `x.permute(1, 0)`. For 4-D you write
`x.permute(0, 2, 1, 3)` and must remember that position `1` is `heads`.
**sempyt:** name the order you want; the library works out the permute:

```python
x.to(col, row)            # transpose by name
```

`to` also splits and merges (next sections) — it is the one tool for all of it.

---

## reshape / view — same numbers, new grouping

**What:** keep the numbers in the same order, but read them as a different shape.
**When:** flattening a feature block before a linear layer.
**Visual:**

```
(2,3)  [0 1 2]     reshape to (6,)     [0 1 2 3 4 5]
       [3 4 5]                   or to (3,2):  [0 1]
                                                [2 3]
                                                [4 5]
```

The numbers march left-to-right, top-to-bottom; reshape just re-chops the line.
**PyTorch:** `x.reshape(6)` / `x.view(3, 2)`. Gotcha: `view` needs the data to be
**contiguous** (see glossary); `reshape` handles it for you.
**sempyt:** you rarely reshape by hand — `to` with a `ProductDim` does it by name.

---

## split — cut one axis into named pieces

**What:** divide one axis into several sub-axes.
**When:** the classic case — a fused QKV projection produces one wide axis that
is really `3 × heads × head_dim`; you split it to get q, k, v.

**Visual (split `col` of size 3 into three size-1 pieces `a`,`b`,`c`):**

```
x (row, col=3)            split col -> a,b,c (size 1 each)
   c0 c1 c2
r0 [0 1 2]      --->   a: [0]   b: [1]   c: [2]     (per row)
r1 [3 4 5]             a: [3]   b: [4]   c: [5]
```

**PyTorch:** `q, k, v = x.split(C, dim=2)` — you must know it is axis `2`.
**sempyt:**

```python
a, b, c = dims("a b c", a=1, b=1, c=1)
q, k, v = sp.split(x, over=col, sizes=(1, 1, 1), names=(a, b, c))
```

---

## merge / `ProductDim` — fuse axes into one

**What:** combine two axes into a single axis whose size is the product.
**When:** heads × head_dim → one `inner` axis before a linear layer; the exact
inverse of split.

**Visual (merge `row`×`col` into `flat` of size 6):**

```
x (row=2, col=3)          to(flat)  where flat = ProductDim(row, col)
   c0 c1 c2
r0 [0 1 2]      --->   flat: [0 1 2 3 4 5]
r1 [3 4 5]
```

**PyTorch:** `x.reshape(2 * 3)` — you compute the product yourself.
**sempyt:** the `ProductDim` knows its factors, so split and merge are exact
inverses with no size bookkeeping:

```python
flat = sp.ProductDim(row, col)
x.to(flat)                # (6,)
x.to(flat).to(row, col)   # back to (2, 3), exactly
```

---

## stack — build a new axis from a list

**What:** take several same-shaped tensors and pile them along a **new** axis.
**When:** you computed each head (or each time-step) separately and want them
back in one tensor.

**Visual (stack two `(row,col)` matrices into a new `heads` axis):**

```
item0        item1                stacked (row, col, heads=2)
[0 1 2]      [10 11 12]           heads=0      heads=1
[3 4 5]      [13 14 15]      r0 [ 0  1  2]   [10 11 12]
                             r1 [ 3  4  5]   [13 14 15]
```

**PyTorch:** `torch.stack([a, b], dim=2)` — position `2` again.
**sempyt:**

```python
heads = Dim("heads")
stacked = sp.stack([x, x + 10], into=heads)        # new axis, size 2
stacked = sp.stack([x, x + 10], into=heads, at=col)  # insert before 'col'
```

---

## unstack — peel an axis back into a list

**What:** the inverse of `stack`; remove an axis and get a Python list of slices.
**When:** after a fused projection, split q/k/v into three separate tensors to
feed them to different code paths.

**Visual:** exactly the `stack` diagram read right-to-left.

**PyTorch:** `a, b = x.unbind(dim=2)` or `x[:, :, 0], x[:, :, 1]`.
**sempyt:**

```python
a, b = sp.unstack(stacked, over=heads)   # list of two (row, col) tensors
```

---

## cat — glue along an existing axis

**What:** join tensors end-to-end along an axis they already share.
**When:** concatenating chunks of a sequence, or features from two branches.

**Visual (cat along `row`):**

```
x (row=2, col=3)          cat([x, x], over=row)  ->  (row=4, col=3)
[0 1 2]                   [0 1 2]
[3 4 5]                   [3 4 5]
                          [0 1 2]
                          [3 4 5]
```

**PyTorch:** `torch.cat([x, x], dim=0)`.
**sempyt:** `sp.cat([x, x], over=row)`. The result gets a fresh `row`-named axis
of the summed size.

---

## select / `x[dim, i]` — pick one slice, drop the axis

**What:** take a single position along an axis; that axis disappears.
**When:** "give me the last token's vector" before sampling.

**Visual:**

```
x (row, col)              select(x, over=row, index=1)  ->  (col,)
[0 1 2]                   [3 4 5]
[3 4 5]                   (row axis is gone)
```

**PyTorch:** `x[1]` or `x.select(0, 1)`.
**sempyt:** `sp.select(x, over=row, index=1)` or the sugar `x[row, 1]`.

---

## take — cut a window out of an axis

**What:** keep a contiguous range `[start, start+length)` of an axis.
**When:** sliding windows, cropping a sequence to the current length.

**Visual:**

```
x (row=2, col=3)          take(over=col, start=1, length=2, new=win)
[0 1 2]                   win: [1 2]
[3 4 5]                          [4 5]
```

**PyTorch:** `x.narrow(1, 1, 2)` or `x[:, 1:3]`.
**sempyt:** `sp.take(x, over=col, start=1, length=2, new=Dim("win"))`.

---

## repeat_along — duplicate along an axis

**What:** repeat each slice of an axis a fixed number of times.
**When:** GQA — you have `kv_heads` key/value heads but `heads` query heads, so
you repeat each KV head to match.

**Visual (repeat `row` twice):**

```
x (row=2, col=3)          repeat_along(over=row, times=2, new=row2)
[0 1 2]                   [0 1 2]
[3 4 5]                   [0 1 2]
                          [3 4 5]
                          [3 4 5]
```

**PyTorch:** `x.repeat_interleave(2, dim=0)`.
**sempyt:** `sp.repeat_along(x, over=row, times=2, new=Dim("row2"))`.

---

## shift — slide an axis, zero-fill the gap

**What:** move every value `by` positions along an axis; the vacated end is
filled with zeros (not wrapped around).
**When:** RWKV "token shift" — each position should see the *previous* token, so
you shift the sequence by one and the first position sees a zero.

**Visual (shift `col` by 1):**

```
x (row, col)              shift(x, over=col, by=1)
[0 1 2]                   [0 0 1]
[3 4 5]                   [0 3 4]     (2 and 5 fall off, 0 fills the left)
```

**PyTorch:** `torch.roll(x, 1, dims=1)` then manually zero the wrapped cell —
easy to forget. **sempyt:** `sp.shift(x, over=col, by=1)` does both.

---

## roll — slide an axis, wrap around

**What:** like `shift`, but values that fall off one end reappear at the other.
**When:** circular convolutions, periodic signals. (Rarely what you want for
causal sequences — that is `shift`.)

**Visual (roll `col` by 1):**

```
x (row, col)              roll(x, over=col, shift=1)
[0 1 2]                   [2 0 1]
[3 4 5]                   [5 3 4]     (2 and 5 wrapped to the front)
```

**PyTorch:** `torch.roll(x, 1, dims=1)`. **sempyt:** `sp.roll(x, over=col, shift=1)`.

---

## flip — reverse an axis

**What:** mirror an axis end-to-end.
**When:** reversing a sequence, symmetric kernels, some convolutions.

**Visual:**

```
x (row, col)              flip(x, over=col)
[0 1 2]                   [2 1 0]
[3 4 5]                   [5 4 3]
```

**PyTorch:** `torch.flip(x, dims=[1])`. **sempyt:** `sp.flip(x, over=col)`.

---

## pad — grow an axis with a fill value

**What:** add `left`/`right` cells of a constant (usually 0) to an axis.
**When:** making variable-length sequences the same length so they batch.

**Visual (pad `col` by 1 on each side):**

```
x (row, col)              pad(x, over=col, left=1, right=1)
[0 1 2]                   [0 0 1 2 0]
[3 4 5]                   [0 3 4 5 0]
```

**PyTorch:** `F.pad(x, (1, 1))` — note the *reversed*, last-axis-first order,
a classic trap. **sempyt:** `sp.pad(x, over=col, left=1, right=1)` names the axis.

---

## The pattern

Every one of these is "name the axis, say what to do with it". PyTorch makes you
translate that into positions (`dim=0`, `dim=1`, `dims=[1]`, `(1, 1)`), and the
translation is where bugs hide. sempyt keeps the name the whole way through.

Next: [03 — Math and reductions](03-math-and-reductions.md).
