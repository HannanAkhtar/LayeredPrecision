import torch

from lp_quantizer_families import quantize_family, dequantize_family
from lp_uniform_calibration import ALPHAS, dequantize_uniform, quantize_uniform


def test_ideal_uniform_midpoint_beats_extrema_grid():
    residual = torch.linspace(-1, 1, 40001).reshape(1, -1)
    _, old = quantize_family(residual, "S")
    u0, middle = quantize_uniform(residual, "U0")
    calibrated, best = quantize_uniform(residual, "U")
    assert abs(u0["scale"].item() - 0.5) < 1e-7
    assert torch.mean((residual - middle)**2) < torch.mean((residual - old)**2)
    assert torch.mean((residual - best)**2) <= torch.mean((residual - middle)**2) + 1e-12
    assert calibrated["alpha_index"].item() == 12
    assert abs(torch.mean((residual - middle)**2).item() - 1/48) < 2e-6
    assert 0.5 in ALPHAS
    assert torch.equal(best, dequantize_uniform(calibrated))


def test_zero_constant_and_small_channels():
    for values in (torch.zeros(2, 7), torch.ones(2, 7),
                   torch.full((2, 7), -2.0), torch.full((2, 7), 1e-35)):
        for family in ("U0", "U"):
            params, decoded = quantize_uniform(values, family)
            assert torch.isfinite(decoded).all()
            assert params["quantized"].dtype == torch.uint8
            assert params["quantized"].min() >= 0 and params["quantized"].max() <= 3
            assert torch.equal(decoded, dequantize_uniform(params))


def test_ties_four_codes_asymmetry_and_determinism():
    values = torch.tensor([[-1.5, -0.5, 0, 0.5, 1.5],
                           [-2.0, -0.25, 0.1, 0.9, 3.0]])
    p, decoded = quantize_uniform(values, "U0")
    assert p["quantized"][0].tolist() == [0, 1, 1, 2, 3]
    assert torch.equal(decoded, dequantize_uniform(p))
    for family in ("U0", "U"):
        a, da = quantize_uniform(values, family)
        b, db = quantize_uniform(values, family)
        assert torch.equal(a["quantized"], b["quantized"])
        assert torch.equal(a["scale"], b["scale"])
        assert torch.equal(da, db)


def test_calibration_never_worse_per_channel():
    torch.manual_seed(23)
    residual = torch.randn(13, 109) * torch.linspace(1e-6, 3, 13)[:, None]
    _, u0 = quantize_uniform(residual, "U0")
    _, u = quantize_uniform(residual, "U")
    mse0 = (residual.double() - u0.double()).square().sum(1)
    mse = (residual.double() - u.double()).square().sum(1)
    assert torch.all(mse <= mse0 + 1e-12)


def test_original_family_reference_definitions_unchanged():
    values = torch.tensor([[-1.0, -0.25, 0.3, 1.0]])
    s, decoded = quantize_family(values, "S")
    assert torch.equal(s["scale"], torch.tensor([[2 / 3]]))
    assert torch.equal(decoded, dequantize_family(s, "S"))
    for family in ("A", "N"):
        one, decoded = quantize_family(values, family)
        two, again = quantize_family(values, family)
        assert torch.equal(one["quantized"], two["quantized"])
        assert torch.equal(decoded, again)
