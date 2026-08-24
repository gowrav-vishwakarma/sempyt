# 04 — Tensor algebra

These are the operations that **match aligned axes against each other** — the
"line up matched staff and combine" family from page 01. They are where named
axes pay off the most, because the whole point is *which* axes line up.

```python
import torch
import sempyt as sp
from sempyt import Dim, dims, named, SplitComplex
```

---

## contract — named matrix multiplication

**What:** matmul, but you name the axis that lines up and gets summed away.
**When:** every linear layer, every attention score (`query · key`), every read
from an associative memory.

**Visual (the page-01 matmul):** `worker×task` times `task×station`, summing
over the shared `task` axis:

```
A (worker, task)   B (task, station)     contract(A, B, over=task)
[1 2 3]            [1 0]                 -> (worker, station)
[4 5 6]       @    [0 1]            =    [ 4  2 ]
                   [1 0]                 [10  5 ]
        the 'task' axis matched and disappeared
```

**PyTorch:** `A @ B`, or in many dims `torch.einsum("bhtd,bhsd->bhts", q, k)` —
a string of single letters you must keep in the right order.
**sempyt:**

```python
worker, task = dims("worker task", worker=2, task=3)
station = Dim("station", 2)
A = named(torch.tensor([[1., 2., 3.], [4., 5., 6.]]), (worker, task))
B = named(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), (task, station))
sp.contract(A, B, over=task)     # (worker, station); sizes checked for you
```

In higher dimensions you keep the un-contracted axes as batch automatically —
`contract(q, k, over=head_dim)` on `(batch, heads, tokens, head_dim)` tensors
gives `(batch, heads, tokens_q, tokens_k)` with no einsum string at all.

---

## read — inner product over one axis

**What:** a dot product generalized to tensors: line up one shared axis and sum
it away. (With complex data it can conjugate the probe first — see `conj`.)
**When:** reading a value out of an associative memory with a query; cosine
similarity.

**Visual:**

```
q (task,)   k (task,)        read(q, k, over=task)
[1 2 3]  ·  [4 5 6]     =    1*4 + 2*5 + 3*6 = 32   (a scalar)
```

```python
sp.read(q, k, over=task)    # 32
```

---

## outer — outer product (build a grid from two vectors)

**What:** the opposite of a dot product. Instead of collapsing an axis, it
**creates** two: every entry of one vector times every entry of the other.
**When:** writing into an outer-product memory (`S += value ⊗ key`).

**Visual:**

```
v (di,)      k (dj,)            outer(v, k, over=(di, dj))
[1]          [1 0 1]            -> (di=2, dj=3)
[2]                             [1] * [1 0 1]   [1 0 1]
                                  [2] * [1 0 1] = [2 0 2]
```

```python
di, dj = Dim("di", 2), Dim("dj", 3)
v = named(torch.tensor([1., 2.]), (di,))
k = named(torch.tensor([1., 0., 1.]), (dj,))
sp.outer(v, k, over=(di, dj))   # [[1, 0, 1], [2, 0, 2]]
```

---

## gather / at — look up rows by integer index

**What:** treat one axis as a table and pull out the rows an integer tensor
points to. **When:** embeddings (token id → its row), LUT lookups, gathering the
top-k winners.

**Visual (look up rows of a table):**

```
table (rows, feat)        idx over 'rows'      at(table, idx, over=rows)
 r0 [0 0]                 batch0 -> r1         batch0: [1 1]
 r1 [1 1]                 batch1 -> r0         batch1: [0 0]
 r2 [2 2]
 r3 [3 3]
```

```python
idx = sp.argmax(scores, over=rows)        # an Index pointing into 'rows'
sp.at(table, idx, over=rows)              # the gathered rows
```

---

## segment / unsegment — fold a long axis into chunks

**What:** chop a long `time` axis into a grid of `(blocks, block_len)`, padding
the tail if needed; `unsegment` is the exact inverse (and crops the padding).
**When:** chunked / block-wise processing of long sequences (linear attention,
scans) where you handle one block at a time.

**Visual (fold `time=5` into `blocks × blen=2`, padding one cell):**

```
x (time=5)                segment -> (blocks=3, blen=2)
 t0 [0]                   block0: [0 1]
 t1 [1]                   block1: [2 3]
 t2 [2]                   block2: [4 0]   <- last cell is padding
 t3 [3]
 t4 [4]
```

```python
blocks, blen = Dim("blocks"), Dim("blen", 2)
chunked, info = sp.segment(x, time=time, into=(blocks, blen))
restored = sp.unsegment(chunked, info)    # exactly back to (time=5)
```

---

## scan — a running recurrence along an axis

**What:** compute `h_t = decay ⊙ h_{t-1} + u_t` for every `t` — each step depends
on the previous. **When:** linear RNNs, SSMs (Mamba), RWKV, EMAs.

**Visual (decay λ=0.5, input all 1s):**

```
u (time,)        h_t = 0.5*h_{t-1} + u_t
[1 1 1 1]   ->   h = [1.0, 1.5, 1.75, 1.875]
                 (1, .5*1+1, .5*1.5+1, .5*1.75+1)
```

```python
time = Dim("time", 4)
u = named(torch.ones(4), (time,))
lam = named(torch.tensor(0.5), ())               # constant decay
sp.scan(u, lam, over=time, mode="recurrent")     # [1.0, 1.5, 1.75, 1.875]
```

`decay` may omit the time axis (constant λ) or include it (per-step γ_t, as in
Mamba). Modes `recurrent` / `chunked` / `parallel` trade speed for memory.

---

## conj — complex conjugate on the pair axis

**What:** sempyt stores complex numbers **split-real** on a named `pair` axis of
size 2: `[real, imag]`. The conjugate flips the sign of the imaginary part.
**When:** complex algebra (phasor/VSA memory, some qllm variants) — multiplying
by the conjugate is how you "unbind" or normalize a complex number.

**Visual:**

```
z (n, pair)              conj(z)
[3 4]          ->        [3 -4]        (real kept, imaginary negated)
```

```python
px = Dim("px", 2)
z = named(torch.tensor([[3., 4.]]), (Dim("n", 1), px), SplitComplex(px))
z.conj()    # [[3, -4]]
```

---

## alias — give an axis a fresh identity

**What:** rename an axis to a **new** `Dim` object. Same numbers, same size, new
name — so it no longer "matches" the old axis in a contraction.
**When:** self-attention needs *two* copies of the sequence axis — one for the
query positions, one for the key positions. If they shared one `Dim`, `contract`
would wrongly try to line them up; aliasing one to `Tk` keeps them distinct.

**Visual:**

```
q (batch, time, d)              q.alias(time, Tq)  ->  (batch, Tq, d)
                                same numbers; the axis is now a distinct object
```

```python
Tq, Tk = Dim("Tq"), Dim("Tk")
scores = sp.contract(q.alias(time, Tq), k.alias(time, Tk), over=d)
```

---

## Dimension-rules cheatsheet

Every rule PyTorch makes you hold in your head, and the sempyt feature that
removes it:

| PyTorch rule you track | sempyt |
|---|---|
| which position is which axis (`dim=1` vs `dim=2`) | name the `Dim` |
| right-aligned broadcasting + manual `unsqueeze` | broadcast over union of named axes |
| `keepdim=True` so a later broadcast still lines up | reductions drop by name; broadcast is positional-free |
| inner sizes must match for `@` | `contract(over=Dim)` checks the named sizes |
| `view` needs `.contiguous()` first | `to(...)` handles layout, returns contiguous via `raw` |
| einsum letter bookkeeping | named axes, no letters |
| two logical uses of one length (q vs k positions) | `alias` to a fresh `Dim` |

---

Next: [05 — Glossary](05-glossary.md).
