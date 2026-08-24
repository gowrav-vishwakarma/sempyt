"""ComplexLinear / ComplexRMSNorm numerics (previously untested)."""

import torch

import sempyt as sp
from sempyt import Dim, SplitComplex, dims, named
from sempyt.nn import ComplexLinear, ComplexRMSNorm


def _split(t: torch.Tensor, layout, pair):
    return named(torch.stack([t.real, t.imag], dim=-1), tuple(layout) + (pair,),
                 SplitComplex(pair))


def test_complex_linear_matches_torch_complex_matmul():
    batch, feat_in, feat_out = dims("batch in out", batch=4, **{"in": 6, "out": 5})
    pair = Dim("complex_pair", 2)
    layer = ComplexLinear(feat_in, feat_out, pair, bias=True)
    x = torch.randn(4, 6, dtype=torch.cfloat)

    got = layer(_split(x, (batch, feat_in), pair))
    weight = torch.complex(layer.weight_real, layer.weight_imag)
    expected = x @ weight.T + torch.complex(layer.bias_real, layer.bias_imag)
    torch.testing.assert_close(sp.real(got).data, expected.real)
    torch.testing.assert_close(sp.imag(got).data, expected.imag)


def test_complex_rmsnorm_preserves_phase_and_normalises_magnitude():
    batch, feature = dims("batch feature", batch=3, feature=8)
    pair = Dim("complex_pair", 2)
    norm = ComplexRMSNorm(feature, pair)
    x = torch.randn(3, 8, dtype=torch.cfloat)

    out = norm(_split(x, (batch, feature), pair))
    got = torch.complex(sp.real(out).data, sp.imag(out).data)
    torch.testing.assert_close(got.angle(), x.angle(), atol=1e-5, rtol=1e-5)
    # scale is all ones at init, so magnitudes come out RMS-normalised
    torch.testing.assert_close(
        got.abs().pow(2).mean(-1).sqrt(),
        torch.ones(3), atol=1e-3, rtol=1e-3,
    )


def test_complex_rmsnorm_is_layout_independent():
    """The normalised axis is found by name, not by position."""
    batch, feature, time = dims("batch feature time", batch=2, feature=4, time=3)
    pair = Dim("complex_pair", 2)
    norm = ComplexRMSNorm(feature, pair)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat)

    canonical = norm(_split(x, (batch, time, feature), pair))
    shuffled = norm(_split(x, (batch, time, feature), pair).to(batch, feature, time, pair))
    torch.testing.assert_close(
        canonical.data, shuffled.to(batch, time, feature, pair).data
    )


def test_complex_rmsnorm_guards_phase_with_the_abs_epsilon():
    """Dividing by |z| must use the abs epsilon, not the larger RMS epsilon."""
    batch, feature = dims("batch feature", batch=1, feature=4)
    pair = Dim("complex_pair", 2)
    norm = ComplexRMSNorm(feature, pair, eps=1e-6)
    x = torch.randn(1, 4, dtype=torch.cfloat)
    z = _split(x, (batch, feature), pair)

    out = norm(z)
    magnitude = (x.real.square() + x.imag.square() + 1e-8).sqrt()
    rms = (magnitude.square().mean(-1, keepdim=True) + 1e-6).sqrt()
    scaled = (magnitude / rms) * norm.scale
    expected = torch.stack([x.real, x.imag], dim=-1) * (
        scaled / (magnitude + 1e-8)
    ).unsqueeze(-1)
    torch.testing.assert_close(out.data, expected)
