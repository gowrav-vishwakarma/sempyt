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

```
v (di=2)      k (dj=3)            outer(v, k, over=(di, dj))
[1]           [1, 0, 1]           Result (2x3)
[2]      X                  =     [1, 0, 1]
                                  [2, 0, 2]
```

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

```
x (time=5)                segment into (blocks=3, blen=2)
 t0 [0]                   block0: [0, 1]
 t1 [1]      --->         block1: [2, 3]
 t2 [2]                   block2: [4, 0] (padded with 0)
 t3 [3]
 t4 [4]
```

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
**Why:** In Self-Attention, we have one sequence. To calculate `query · key`, we need to match "query positions" against "key positions". If both axes are just named `time`, `contract` will try to sum them away!

```python
import torch
import sempyt as sp
from sempyt import dims, named

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
