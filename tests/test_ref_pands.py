"""pands v1 memory three-mode agreement + v2 PhaseLUT / diag_scan."""

import math

import torch

from sempyt import Dim, SplitComplex, named
from sempyt.nn import PhaseLUT
from sempyt.testing import assert_close
from sempyt.zoo.pands_vps import OuterProductBand, diag_scan_named


def test_outer_product_three_modes_agree():
    d, H, dh = Dim("d", 16), Dim("H", 2), Dim("dh", 8)
    B, T = Dim("B", 2), Dim("T", 10)
    band = OuterProductBand(d, H, dh, decay_bias=-2.0)
    x = named(torch.randn(2, 10, 16), (B, T, d))
    y_r, s_r = band.recurrent(x, T, B)
    y_p, s_p = band.parallel(x, T, B)
    y_c, s_c = band.chunked(x, T, B, chunk=4)
    assert_close(y_p.data, y_r.data, rtol=1e-4, atol=1e-5, msg="parallel vs recurrent y")
    assert_close(y_c.data, y_r.data, rtol=1e-4, atol=1e-5, msg="chunked vs recurrent y")
    assert_close(s_p.data, s_r.data, rtol=1e-4, atol=1e-5, msg="parallel vs recurrent S")
    assert_close(s_c.data, s_r.data, rtol=1e-4, atol=1e-5, msg="chunked vs recurrent S")


def test_phase_lut_matches_trig():
    bins, px = Dim("bins", 256), Dim("px", 2, role="complex_pair")
    lut = PhaseLUT(bins, px, phase_mode="lut")
    theta = named(torch.tensor([0.0, math.pi / 2, math.pi]), (Dim("n", 3),))
    z_exact = lut.from_angle(theta, mode="exact")
    z_lut = lut.from_angle(theta, mode="lut")
    assert_close(z_exact.data, z_lut.data, rtol=1e-2, atol=1e-2)


def test_diag_scan_named_matches_recurrent_loop():
    B, T, d, px = Dim("B", 2), Dim("T", 9), Dim("d", 4), Dim("px", 2)
    pol = SplitComplex(px)
    u = named(torch.randn(2, 9, 4, 2), (B, T, d, px), pol)
    # |λ| < 1
    ang = torch.rand(4) * 2 * math.pi
    mag = torch.rand(4) * 0.5 + 0.2
    lam = named(torch.stack([mag * ang.cos(), mag * ang.sin()], dim=-1), (d, px), pol)
    rec = diag_scan_named(u, lam, T, mode="recurrent")
    chk = diag_scan_named(u, lam, T, mode="chunked", chunk=4)
    # raw python loop
    h = u.data.new_zeros(2, 4, 2)
    outs = []
    for t in range(9):
        h = torch.stack(
            [
                lam.data[..., 0] * h[..., 0] - lam.data[..., 1] * h[..., 1],
                lam.data[..., 0] * h[..., 1] + lam.data[..., 1] * h[..., 0],
            ],
            dim=-1,
        ) + u.data[:, t]
        outs.append(h)
    exp = torch.stack(outs, dim=1)
    assert_close(rec.to(B, T, d, px).data, exp, rtol=1e-4, atol=1e-5)
    assert_close(chk.to(B, T, d, px).data, exp, rtol=1e-4, atol=1e-5)
