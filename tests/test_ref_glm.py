"""GLM/Kimi-style block: builds, forwards, backprops."""

import torch

from sempyt import Dim, named
from sempyt.zoo.glm import GLMBlock, GLMConfig


def test_glm_block_forward_backward():
    cfg = GLMConfig()
    block = GLMBlock(cfg)
    B, T = Dim("B", 2), Dim("T", 7)
    x = named(torch.randn(2, 7, cfg.d_model, requires_grad=True), (B, T, block.C))
    y = block(x, B, T)
    assert y.data.shape == (2, 7, cfg.d_model)
    y.data.sum().backward()
    assert x.data.grad is not None
    assert block.q.weight.grad is not None
    assert block.router.weight.grad is not None
