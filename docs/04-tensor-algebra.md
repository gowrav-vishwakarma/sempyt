# 04 — Tensor Algebra

These operations are the heart of neural networks. They **match axes against each other** to combine information. sempyt makes this incredibly safe by using Dim names instead of position math.

---

## `contract` — Named Matrix Multiplication

**What:** Matrix multiplication, but you name the axis that "lines up" and disappears.
**Analogy:** A factory where "Workers" meet "Tasks", and "Tasks" meet "Stations". The "Tasks" axis lines up and vanishes, leaving a "Worker x Station" result.

**Visual:**

| A (`worker=2, task=3`) | B (`task=3, station=2`) | `contract(A, B, over=task)` |
| :--- | :--- | :--- |
| `[1, 2, 3]` <br> `[4, 5, 6]` | `[1, 0]` <br> `[0, 1]` <br> `[1, 0]` | `[4, 2]` <br> `[10, 5]` |
| | (Matches on `task`) | (Result: `worker x station`) |

```python
import torch
import sempyt as sp
from sempyt import dims, named

# Setup axes
w, t, s = dims("worker task station", worker=2, task=3, station=2)

# Create data
A = named(torch.tensor([[1., 2., 3.], [4., 5., 6.]]), (w, t))
B = named(torch.tensor([[1., 0.], [0., 1.], [1., 0.]]), (t, s))

# Match the 'task' axis and sum it away. 
# Result is automatically (worker, station)
C = sp.contract(A, B, over=t)
```

---

## Associative Memory (`outer` and `read`)

These are the two basic moves for "writing" to and "reading" from an associative memory.

### 1. `outer` (The Write)
Takes two lists and makes a grid. Every item in list A meets every item in list B.

| v (`di=2`) | value |
|------------|-------|
| i0         | 1     |
| i1         | 2     |

| k (`dj=3`) | d0 | d1 | d2 |
|------------|----|----|-----|
|            | 1  | 0  | 1   |

| `outer(v, k)` (`di×dj`) | d0 | d1 | d2 |
|-------------------------|----|----|-----|
| i0                      | 1  | 0  | 1   |
| i1                      | 2  | 0  | 2   |

Each cell is `v[i] * k[j]` — all pairs at once.

### 2. `read` (The Read)
The reverse: take a grid and a query list. "Look up" the value by matching axes and summing.

```python
import torch
import sempyt as sp
from sempyt import Dim, named

di, dj = Dim("di", 2), Dim("dj", 3)
v = named(torch.tensor([1., 2.]), (di,))
k = named(torch.tensor([1., 0., 1.]), (dj,))

# WRITE: create a memory grid from two lists
memory = sp.outer(v, k, over=(di, dj))

# READ: dot product the memory with a query list over dj
query = named(torch.tensor([1., 0., 1.]), (dj,))
result = sp.read(memory, query, over=dj) # Result layout: (di,)
```

---

## `at` — The Named Table Lookup

**What:** Look up rows in a table using integer IDs.
**When:** Word embeddings (token ID → feature vector).

**Visual:**

| Table (`rows=4, features=2`) | IDs (Index over `rows`) | `at(Table, IDs)` |
| :--- | :--- | :--- |
| `r0: [0, 0]` <br> `r1: [1, 1]` <br> `r2: [2, 2]` <br> `r3: [3, 3]` | `[1, 0]` | `[1, 1]` <br> `[0, 0]` |

```python
import torch
import sempyt as sp
from sempyt import dims, named

# Setup table
rows, feat = dims("rows features", rows=4, feat=2)
table = named(torch.tensor([[0.,0.], [1.,1.], [2.,2.], [3.,3.]]), (rows, feat))

# Integers pointing to row 1 and row 0
ids = torch.tensor([1, 0])

# Pull out those exact rows by naming the 'rows' axis
result = sp.at(table, ids, over=rows)
```

---

## `segment` — Folding Time into Blocks

**What:** Chop a long sequence into blocks.
**When:** Handling long context in chunks.

| time | value |
|------|-------|
| t0   | 0     |
| t1   | 1     |
| t2   | 2     |
| t3   | 3     |
| t4   | 4     |

Folded into `(blocks=3, blen=2)` — tree view:

| block | slot | value |
|-------|------|-------|
| b0    | s0   | 0     |
|       | s1   | 1     |
| b1    | s0   | 2     |
|       | s1   | 3     |
| b2    | s0   | 4     |
|       | s1   | 0     |

Last block is **padded** with 0 because `time=5` is not divisible by `blen=2`.

```python
import torch
import sempyt as sp
from sempyt import Dim, named

t = Dim("time", 5)
x = named(torch.arange(5).float(), (t,))

# Chop time into blocks of length 2
blocks, blen = Dim("blocks"), Dim("blen", 2)
chunked, info = sp.segment(x, time=t, into=(blocks, blen))
```

---

## `scan` — The Running Total

**What:** The core of RNNs and Mamba. Each step is `h_t = decay * h_{t-1} + input_t`.
**Visual:**

| input `u` | decay `lam` | Result `h` |
| :--- | :--- | :--- |
| `[1, 1, 1, 1]` | `0.5` | `[1.0, 1.5, 1.75, 1.875]` |

```python
import torch
import sempyt as sp
from sempyt import Dim, named

time = Dim("time", 4)
u = named(torch.ones(4), (time,))
decay = named(torch.tensor(0.5), ())

# Calculate the recurrence along the 'time' axis
h = sp.scan(u, decay, over=time)
```

---

## `alias` — The "Identity" Switch

**What:** Rename a dimension so it doesn't accidentally "match" itself.
**Why:** In Self-Attention we have one physical sequence, but we conceptually need two views of it: a *query* position and a *key* position. If both Q and K keep the same `time` Dim object, `contract(..., over=d)` collapses the result to a single `time` axis — the query and key positions become the same thing, and the attention matrix is silently wrong.

### The silent bug without `alias`

Take a tiny 4-token sequence:

```python
import torch
import sempyt as sp
from sempyt import dims, named

t, d = dims("time d", time=4, d=3)

q = named(torch.tensor([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1]]).float(), (t, d))

k = named(torch.tensor([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 1, 1]]).float(), (t, d))
```

If you write:

```python
scores = sp.contract(q, k, over=d)
```

`contract` sees that `t` is the **same Dim object** in both Q and K, so it keeps only one `time` axis. The result is:

```python
scores.shape == (t,)   # -> (4,)
```

It computed only the diagonal dot products: position 0 with position 0, position 1 with 1, and so on. Query position 0 can **never** attend to key position 1, 2, or 3. The code runs, but the attention matrix is missing.

### How `alias` fixes it

Give the query copy a new identity `t_q` without moving any data:

```python
from sempyt import Dim

tq = Dim("t_q", 4)
scores = sp.contract(q.alias(t, tq), k, over=d)
```

Now `t_q` and `t` are different Dim objects, so the result keeps both:

```python
scores.shape == (tq, t)   # -> (4, 4)
```

This is the full attention score matrix: every query position attends to every key position.

**2D result** — attention scores `(t_q=4, time=4)` after `contract(q.alias(t,tq), k, over=d)`:

Each cell is "how much does query position `t_q` attend to key position `time`?"

| t_q | t0 | t1 | t2 | t3 |
|-----|----|----|----|-----|
| q0  | ·  | ·  | ·  | ·   |
| q1  | ·  | ·  | ·  | ·   |
| q2  | ·  | ·  | ·  | ·   |
| q3  | ·  | ·  | ·  | ·   |

**4D before contraction** — batched multi-head Q and K: `(batch=2, heads=2, time=4, d=64)`.
`alias` only renames one axis on Q; `contract(..., over=d)` sums away `d` and leaves
`(batch, heads, t_q, time)` — a **folder of score matrices per head per batch**:

| batch | head | t_q | t0 | t1 | t2 | t3 |
|-------|------|-----|----|----|----|-----|
| b0    | h0   | q0  | ·  | ·  | ·  | ·   |
|       |      | q1  | ·  | ·  | ·  | ·   |
|       | h1   | q0  | ·  | ·  | ·  | ·   |
|       |      | q1  | ·  | ·  | ·  | ·   |
| b1    | h0   | q0  | ·  | ·  | ·  | ·   |
|       |      | q1  | ·  | ·  | ·  | ·   |
|       | h1   | q0  | ·  | ·  | ·  | ·   |
|       |      | q1  | ·  | ·  | ·  | ·   |

(Shown with `time=2` query rows for space; real models use longer sequences.)

```python
import torch
import sempyt as sp
from sempyt import dims, named, Dim

# One sequence of length 4
t, d = dims("time d", time=4, d=64)
q = named(torch.randn(4, 64), (t, d))
k = named(torch.randn(4, 64), (t, d))

# Give the query copy a new name 't_q' so it stays distinct from 'time'
tq = Dim("t_q", 4)
scores = sp.contract(q.alias(t, tq), k, over=d)

# Result layout is (t_q, time) -> 4x4 scores matrix
```

---

Next: [05 — Glossary](05-glossary.md).
