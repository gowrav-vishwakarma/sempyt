# 05 — Glossary

A quick reference for the terms used in this guide.

---

### Basic Structures
- **Scalar**: A single number (Rank 0).
- **Vector**: A list of numbers (Rank 1).
- **Matrix**: A 2D grid of numbers (Rank 2).
- **Tensor**: The general term for a grid of any rank (3D, 4D, etc.).
- **Rank / ndim**: The number of axes a tensor has.
- **Shape**: The size of each axis (e.g., `(batch, time, features)`).

### Axis Concepts
- **Axis / Dim**: One labeled edge of the grid.
- **ProductDim**: A fused axis that is the product of others (e.g., `Heads * Features`).
- **Layout**: The ordered list of `Dim` objects for a tensor.

### Memory & Layout
- **Contiguous**: When numbers are stored in one long line in memory in the "logical" order.
- **Stride**: The "jump" size required to move to the next item along an axis in memory.
- **View / Reshape**: Reading the same memory as a different shape without moving the numbers.

### Core Operations
- **Element-wise**: Doing the same math to every cell independently.
- **Broadcasting**: Stretching a smaller tensor to match a larger one automatically.
- **Reduction**: Shrinking an axis away (e.g., Sum or Mean).
- **keepdim**: A PyTorch flag to keep a collapsed axis as size 1. sempyt avoids this by naming.
- **Contract (Matmul)**: Matching one axis between two tensors and summing it away.
- **Inner Product (Dot)**: Multiplying two lists item-by-item and summing.
- **Outer Product**: Multiplying two lists to create a grid of all pairs.
- **Scan**: A running total or recurrence along an axis.

---

Back to the [index](README.md).
